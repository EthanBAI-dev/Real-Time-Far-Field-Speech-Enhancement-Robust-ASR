"""CRN-Lite —— 因果卷积循环网络语音增强模型。

自行定义而不是引用外部模型仓库，理由有三：
1. 作品集里必须能逐行讲清楚每个设计选择，抄来的代码做不到；
2. **流式 ONNX 导出对模型结构有硬约束**（下面详述），拿现成模型往往要大改；
3. 不引入额外依赖，Colab 上 clone 本仓库就能训。

架构
----
::

    带噪复数谱 (B,2,T,257)
        │  幅度做 0.3 次幂压缩（见 compress()）
    编码器 5 层因果 Conv2d，kernel=(2,5)，频率轴 stride 2
        │  257 → 129 → 65 → 33 → 17 → 9
        │  通道 2 → 16 → 32 → 32 → 48 → 64
    瓶颈 2 层单向 GRU（输入 64×9=576）
        │
    解码器 5 层 ConvTranspose2d，kernel=(1,5)，与编码器逐层跳跃连接
        │  9 → 17 → 33 → 65 → 129 → 257
    输出 有界复数比值掩码 CRM (B,2,T,257)

因果性是硬约束
--------------
- **时间轴只在左侧补零**。任何一处对称 padding 都等于偷看未来帧，
  流式导出立刻失效，而离线指标却依然好看 —— 这是最危险的一类 bug。
- **GRU 必须单向**。双向 GRU 需要整段序列，根本无法流式。
- **解码器时间核取 1**。这是个刻意的取舍：把全部时序建模交给编码器（5 帧感受野）
  和 GRU（无限历史），换来解码器在时间上完全无状态。
  代价是少了一点解码端的时序平滑能力，收益是流式导出时**只需缓存编码器的 5 个状态**，
  解码器一个 cache 都不用 —— 导出图更简单，出错面小得多。

为什么用复数比值掩码（CRM）而不是幅度掩码
------------------------------------------
幅度掩码只能改幅度、沿用带噪相位，这是所有传统 DSP 方法的固有上限
（见 ``rtse/dsp/enhancers.py``）。低 SNR 下相位误差是可懂度损失的主要来源之一。
CRM 同时修正实部与虚部，是本模型**有可能超越 MMSE-LSA 的根本原因**。
指标表里"NN vs DSP"这一栏能不能拉开差距，很大程度上取决于这个选择。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CRNConfig", "CRNLite", "compress", "decompress", "apply_crm", "build_model", "PRESETS"]


@dataclass
class CRNConfig:
    n_freq: int = 257
    enc_channels: tuple[int, ...] = (16, 32, 32, 48, 64)
    gru_hidden: int = 128
    gru_layers: int = 2
    kernel_freq: int = 5
    kernel_time: int = 2
    """编码器的时间核。2 表示每层看当前帧 + 前一帧，感受野 = 层数 + 1 帧。"""
    stride_freq: int = 2
    compress_power: float = 0.3
    """幅度压缩指数。0.3 接近人耳的响度感知律，也是 DNS Challenge 基线的常用取值。"""
    mask_bound: float = 3.0
    """CRM 的幅度上界。不设界的话模型可以输出任意大的掩码，
    低 SNR 帧上会放大出爆音；设太小又限制了对欠估计谱的补偿能力。"""

    @property
    def freq_dims(self) -> list[int]:
        """各层输出的频率维度，编码器与解码器共用（解码器逆序）。"""
        dims, f = [self.n_freq], self.n_freq
        pad = self.kernel_freq // 2
        for _ in self.enc_channels:
            f = (f + 2 * pad - self.kernel_freq) // self.stride_freq + 1
            dims.append(f)
        return dims


PRESETS: dict[str, CRNConfig] = {
    # 主力模型：目标 < 1M 参数
    "crn-lite": CRNConfig(),
    # 极低算力档：用来占据指标表里"参数量最小"那一格，与 DSP 基线正面比较
    "crn-nano": CRNConfig(enc_channels=(8, 16, 16, 24, 32), gru_hidden=64, gru_layers=1),
    # 高性能档：看清楚"堆容量能换到多少指标"，为帕累托前沿提供上界
    "crn-large": CRNConfig(enc_channels=(24, 48, 64, 64, 96), gru_hidden=256, gru_layers=2),
}


def compress(mag: torch.Tensor, power: float = 0.3) -> torch.Tensor:
    """幅度谱的幂律压缩。

    语音谱的动态范围有 60+ dB，直接喂原始幅度会让网络几乎只关注少数高能量频点，
    而语音可懂度恰恰高度依赖低能量的辅音区。压缩后动态范围收窄，
    弱分量得到与其感知重要性相称的权重。
    """
    return mag.clamp_min(1e-8) ** power


def decompress(mag: torch.Tensor, power: float = 0.3) -> torch.Tensor:
    return mag.clamp_min(0.0) ** (1.0 / power)


def apply_crm(spec: torch.Tensor, mask: torch.Tensor, bound: float = 3.0) -> torch.Tensor:
    """把有界复数比值掩码作用到带噪复数谱上。

    Args:
        spec: ``(B, 2, T, F)``，通道 0/1 为实部/虚部。
        mask: ``(B, 2, T, F)``，网络的裸输出。
    """
    # 用 tanh 限幅而不是硬 clamp：硬 clamp 在边界处梯度为 0，
    # 一旦掩码被推到界上就再也学不回来（梯度死区）。
    m = bound * torch.tanh(mask / bound)
    mr, mi = m[:, 0], m[:, 1]
    sr, si = spec[:, 0], spec[:, 1]
    return torch.stack([sr * mr - si * mi, sr * mi + si * mr], dim=1)


class _EncBlock(nn.Module):
    """因果编码器块：左侧补零的 Conv2d + BN + PReLU。"""

    def __init__(self, cin: int, cout: int, cfg: CRNConfig) -> None:
        super().__init__()
        self.pad_t = cfg.kernel_time - 1  # 只在时间轴**左侧**补，右侧不补 → 严格因果
        self.pad_f = cfg.kernel_freq // 2
        self.conv = nn.Conv2d(
            cin, cout,
            kernel_size=(cfg.kernel_time, cfg.kernel_freq),
            stride=(1, cfg.stride_freq),
            padding=(0, self.pad_f),
        )
        self.norm = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(F.pad(x, (0, 0, self.pad_t, 0)))))

    def forward_stream(self, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """流式单帧前向。

        Args:
            x: ``(B, C, 1, F)`` 当前帧。
            cache: ``(B, C, pad_t, F)`` 之前 ``pad_t`` 帧。
        Returns:
            ``(输出帧, 新 cache)``
        """
        inp = torch.cat([cache, x], dim=2)
        new_cache = inp[:, :, 1:]  # 丢掉最老的一帧
        return self.act(self.norm(self.conv(inp))), new_cache


class _DecBlock(nn.Module):
    """解码器块：时间核为 1 的 ConvTranspose2d，因此在时间上完全无状态。"""

    def __init__(self, cin: int, cout: int, cfg: CRNConfig, last: bool = False) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            cin, cout,
            kernel_size=(1, cfg.kernel_freq),
            stride=(1, cfg.stride_freq),
            padding=(0, cfg.kernel_freq // 2),
        )
        self.last = last
        if not last:
            self.norm = nn.BatchNorm2d(cout)
            self.act = nn.PReLU(cout)

    def forward(self, x: torch.Tensor, target_f: int) -> torch.Tensor:
        y = self.conv(x)
        # 转置卷积的输出频率维可能比目标多 1（取决于 output_padding），
        # 直接裁到目标长度比去凑 output_padding 参数更不容易出错
        if y.shape[-1] > target_f:
            y = y[..., :target_f]
        elif y.shape[-1] < target_f:
            y = F.pad(y, (0, target_f - y.shape[-1]))
        return y if self.last else self.act(self.norm(y))


class CRNLite(nn.Module):
    """因果 CRN 语音增强模型。输入输出都是复数谱 ``(B, 2, T, F)``。"""

    def __init__(self, cfg: CRNConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CRNConfig()
        c = self.cfg
        dims = c.freq_dims

        chans = (2,) + c.enc_channels
        self.encoder = nn.ModuleList(
            [_EncBlock(chans[i], chans[i + 1], c) for i in range(len(c.enc_channels))]
        )

        self.bottleneck_f = dims[-1]
        gru_in = c.enc_channels[-1] * self.bottleneck_f
        self.gru = nn.GRU(gru_in, c.gru_hidden, c.gru_layers, batch_first=True)
        self.gru_proj = nn.Linear(c.gru_hidden, gru_in)

        # 解码器：输入通道数翻倍（跳跃连接用 concat 而非相加 ——
        # concat 让网络自己决定如何加权编码器特征，相加是强行等权）
        dec = []
        rev = list(c.enc_channels[::-1])
        for i, cin in enumerate(rev):
            cout = rev[i + 1] if i + 1 < len(rev) else 2
            dec.append(_DecBlock(cin * 2, cout, c, last=(i + 1 == len(rev))))
        self.decoder = nn.ModuleList(dec)
        self.target_dims = dims[::-1][1:]  # 解码器各层的目标频率维

    # ------------------------------------------------------------------ 整段

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Args: spec ``(B, 2, T, F)`` 带噪复数谱。Returns: 增强后复数谱，同形状。"""
        x = self._to_feature(spec)

        skips = []
        for blk in self.encoder:
            x = blk(x)
            skips.append(x)

        b, ch, t, f = x.shape
        g, _ = self.gru(x.permute(0, 2, 1, 3).reshape(b, t, ch * f))
        x = self.gru_proj(g).reshape(b, t, ch, f).permute(0, 2, 1, 3)

        for i, blk in enumerate(self.decoder):
            x = blk(torch.cat([x, skips[-1 - i]], dim=1), self.target_dims[i])

        return apply_crm(spec, x, self.cfg.mask_bound)

    def _to_feature(self, spec: torch.Tensor) -> torch.Tensor:
        """复数谱 → 压缩后的实虚特征。

        压缩作用在**幅度**上、相位保持不变，而不是分别对实虚部做幂运算 ——
        后者会把相位也一起扭曲。
        """
        mag = torch.linalg.vector_norm(spec, dim=1, keepdim=True).clamp_min(1e-8)
        return spec / mag * compress(mag, self.cfg.compress_power)

    # ------------------------------------------------------------------ 流式

    def init_state(self, batch: int = 1, device=None, dtype=torch.float32) -> list[torch.Tensor]:
        """初始化流式状态：5 个编码器 cache + 1 个 GRU 隐状态。"""
        c, dims = self.cfg, self.cfg.freq_dims
        dev = device or next(self.parameters()).device
        chans = (2,) + c.enc_channels
        caches = [
            torch.zeros(batch, chans[i], c.kernel_time - 1, dims[i], device=dev, dtype=dtype)
            for i in range(len(c.enc_channels))
        ]
        h = torch.zeros(c.gru_layers, batch, c.gru_hidden, device=dev, dtype=dtype)
        return caches + [h]

    def forward_stream(
        self, spec_frame: torch.Tensor, state: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """单帧流式前向。

        Args:
            spec_frame: ``(B, 2, 1, F)`` 当前帧的带噪复数谱。
            state: ``init_state()`` 或上一次调用返回的状态。
        Returns:
            ``(增强后的当前帧, 新状态)``

        这个函数与 ``forward`` 的数值一致性由
        ``tests/test_model_streaming.py`` 强制。不一致就说明有地方偷看了未来帧。
        """
        n_enc = len(self.encoder)
        caches, h = list(state[:n_enc]), state[n_enc]

        x = self._to_feature(spec_frame)
        skips, new_caches = [], []
        for i, blk in enumerate(self.encoder):
            x, nc = blk.forward_stream(x, caches[i])
            new_caches.append(nc)
            skips.append(x)

        b, ch, _, f = x.shape
        g, new_h = self.gru(x.permute(0, 2, 1, 3).reshape(b, 1, ch * f), h)
        x = self.gru_proj(g).reshape(b, 1, ch, f).permute(0, 2, 1, 3)

        for i, blk in enumerate(self.decoder):
            x = blk(torch.cat([x, skips[-1 - i]], dim=1), self.target_dims[i])

        return apply_crm(spec_frame, x, self.cfg.mask_bound), new_caches + [new_h]

    # ------------------------------------------------------------------ 统计

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> dict:
        return {
            "params": self.count_params(),
            "params_M": round(self.count_params() / 1e6, 4),
            "enc_channels": list(self.cfg.enc_channels),
            "gru": f"{self.cfg.gru_layers}x{self.cfg.gru_hidden}",
            "freq_dims": self.cfg.freq_dims,
            "receptive_field_frames": len(self.encoder) * (self.cfg.kernel_time - 1) + 1,
            "causal": True,
        }


def build_model(name: str = "crn-lite") -> CRNLite:
    if name not in PRESETS:
        raise KeyError(f"未知模型 {name!r}，可选：{sorted(PRESETS)}")
    return CRNLite(PRESETS[name])
