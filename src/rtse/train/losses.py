"""损失函数。

三项加权组合，各自负责一个不同的失效模式 —— 只用其中任何一项都会有明显短板：

1. **压缩谱复数 MSE**：主损失。在幂律压缩域上同时约束实部与虚部，
   等价于同时监督幅度和相位。这是 DNS Challenge 基线的标准做法。
   单独用它的问题：MSE 对整体尺度敏感，容易学出"整体压低音量"这种偷懒解。

2. **多分辨率 STFT 幅度损失**：用不同窗长（短窗看瞬态、长窗看谐波）约束幅度。
   补的是主损失单一分辨率的盲区 —— 512 点窗对辅音这类短瞬态的时间分辨率不够，
   模型可以把辅音抹平而 loss 几乎不涨。

3. **SI-SDR**：时域、尺度不变。它直接堵住第 1 项的偷懒解 ——
   整体压低音量对 SI-SDR 完全没有收益。

权重默认 1.0 / 0.5 / 0.2。SI-SDR 权重刻意小：它是数值范围很大的 dB 量，
权重给高会主导梯度，把模型推向"SI-SDR 好听感差"（谱减法那种）的方向。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CompressedSpecLoss", "MultiResSTFTLoss", "si_sdr_loss", "CombinedLoss"]

_EPS = 1e-8


def _complex_mag(spec: torch.Tensor) -> torch.Tensor:
    """``(B,2,T,F)`` → ``(B,T,F)`` 幅度。"""
    return torch.linalg.vector_norm(spec, dim=1).clamp_min(_EPS)


class CompressedSpecLoss(nn.Module):
    """幂律压缩域上的复数 + 幅度 MSE。

    分成"复数项"和"纯幅度项"两部分并各占一半，是有意为之：
    只用复数项时，相位误差会在幅度很小的频点上产生很大的梯度
    （那些点的相位本来就没有意义），训练不稳定。
    加入纯幅度项相当于给了一个稳定的、与相位无关的信号。
    """

    def __init__(self, power: float = 0.3, mag_weight: float = 0.5) -> None:
        super().__init__()
        self.power = power
        self.mag_weight = mag_weight

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        est_mag, ref_mag = _complex_mag(est), _complex_mag(ref)
        est_c, ref_c = est_mag**self.power, ref_mag**self.power

        # 压缩后的复数谱：保持相位，只缩放幅度
        est_cplx = est / est_mag.unsqueeze(1) * est_c.unsqueeze(1)
        ref_cplx = ref / ref_mag.unsqueeze(1) * ref_c.unsqueeze(1)

        return (1 - self.mag_weight) * F.mse_loss(est_cplx, ref_cplx) + self.mag_weight * F.mse_loss(
            est_c, ref_c
        )


class MultiResSTFTLoss(nn.Module):
    """多分辨率 STFT 幅度损失（时域信号上计算）。

    三组窗长覆盖不同的时频权衡：
    - 256/128：时间分辨率高，抓瞬态与辅音
    - 512/256：与主管线一致
    - 1024/256：频率分辨率高，抓谐波结构与基频
    """

    def __init__(self, ffts: tuple[int, ...] = (256, 512, 1024)) -> None:
        super().__init__()
        self.ffts = ffts
        for n in ffts:
            self.register_buffer(f"win{n}", torch.hann_window(n), persistent=False)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        loss = est.new_zeros(())
        for n in self.ffts:
            win = getattr(self, f"win{n}")
            kw = dict(n_fft=n, hop_length=n // 4, win_length=n, window=win, return_complex=True)
            e = torch.stft(est, center=True, **kw).abs().clamp_min(_EPS)
            r = torch.stft(ref, center=True, **kw).abs().clamp_min(_EPS)
            # 谱收敛项（相对误差）+ 对数幅度项（绝对误差）。前者对强分量敏感、
            # 后者对弱分量敏感，两者互补是多分辨率损失的标准配方。
            sc = torch.norm(r - e, p="fro") / torch.norm(r, p="fro").clamp_min(_EPS)
            mag = F.l1_loss(torch.log(e), torch.log(r))
            loss = loss + sc + mag
        return loss / len(self.ffts)


def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """负 SI-SDR（越小越好）。与 ``rtse.metrics.intrusive.si_sdr`` 的定义完全一致。"""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + _EPS)
    target = alpha * ref
    noise = est - target
    return -(10 * torch.log10((target.pow(2).sum(-1) + _EPS) / (noise.pow(2).sum(-1) + _EPS))).mean()


class CombinedLoss(nn.Module):
    """三项加权组合。``forward`` 同时返回总损失和各分项，便于在训练曲线上分开看。

    分开记录很重要：训练后期常出现"总 loss 还在降但 SI-SDR 已经不动了"的情况，
    只看总 loss 完全看不出来。
    """

    def __init__(
        self, w_spec: float = 1.0, w_mrstft: float = 0.5, w_sisdr: float = 0.2, power: float = 0.3
    ) -> None:
        super().__init__()
        self.spec = CompressedSpecLoss(power)
        self.mrstft = MultiResSTFTLoss()
        self.w = (w_spec, w_mrstft, w_sisdr)

    def forward(
        self,
        est_spec: torch.Tensor,
        ref_spec: torch.Tensor,
        est_wav: torch.Tensor,
        ref_wav: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_spec = self.spec(est_spec, ref_spec)
        l_mr = self.mrstft(est_wav, ref_wav)
        l_si = si_sdr_loss(est_wav, ref_wav)
        total = self.w[0] * l_spec + self.w[1] * l_mr + self.w[2] * l_si
        return total, {
            "loss": total.item(),
            "spec": l_spec.item(),
            "mrstft": l_mr.item(),
            "si_sdr": -l_si.item(),  # 记正的 SI-SDR，读起来直观
        }
