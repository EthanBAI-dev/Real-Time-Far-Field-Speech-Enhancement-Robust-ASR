"""训练数据集：在线动态混音。

**为什么在线混音而不是预先合成好一批固定的带噪文件**：
预合成的数据集，模型在第 2 个 epoch 就会开始记住"这条语音配这段噪声在这个 SNR"。
在线混音让每个 epoch 见到的组合都不同，等价于把数据量放大了
（语音数 × 噪声数 × SNR 档数 × RIR 数）倍，对小规模子集尤其关键 ——
而我们在 Colab 上恰恰只能用小规模子集。

代价是 CPU 端的混音开销。用 DataLoader 的多 worker 就能盖掉，
实测在 Colab 上不会成为瓶颈（GPU 前向反向远慢于混音）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rtse import SAMPLE_RATE
from rtse.audio.io import read_audio
from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig, num_frames
from rtse.data.synth import apply_rir, mix_at_snr

__all__ = ["MixConfig", "OnlineMixDataset", "collate_specs", "stft_torch"]

AUDIO_EXT = ("*.wav", "*.flac", "*.WAV", "*.FLAC")


@dataclass
class MixConfig:
    segment_seconds: float = 4.0
    snr_range: tuple[float, float] = (-5.0, 20.0)
    reverb_prob: float = 0.5
    """施加混响的概率。不是 100% —— 训练集里必须保留一部分近场干净样本，
    否则模型会把"去混响"当成无条件要做的事，在本来就没混响的输入上产生过处理。"""
    noise_prob: float = 0.98
    """极少量样本不加噪。让模型见过"输入已经很干净"的情况，学会此时少动手。"""
    sample_rate: int = SAMPLE_RATE


def _scan(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    out: list[Path] = []
    for pat in AUDIO_EXT:
        out.extend(root.rglob(pat))
    return sorted(set(out))


def _as_files(spec: str | Path | list, what: str, allow_empty: bool = False) -> list[Path]:
    """把"目录或文件列表"统一成 Path 列表。"""
    files = [Path(p) for p in spec] if isinstance(spec, (list, tuple)) else _scan(spec)
    if not files and not allow_empty:
        raise FileNotFoundError(f"{what}为空：{spec if not isinstance(spec, (list, tuple)) else '给定的列表'}")
    return files


class OnlineMixDataset(Dataset):
    """在线混音数据集，产出 ``(带噪波形, 干净参考波形)`` 对。

    Args:
        clean_dir: 干净语音目录（递归扫描）。
        noise_dir: 噪声目录。
        rir_dir: 房间冲激响应目录。为 None 时用合成 RIR。
        length: 一个 epoch 的样本数。与文件数解耦 ——
            在线混音下"一个 epoch"是人为定义的，按步数控制训练进度更方便。

    **参考信号的选择**：加了混响时，参考是**混响后的干净语音**而不是原始干信号。
    理由与文件实验台一致：让降噪模型为"没能去掉混响"扣分，
    等于逼它同时学两件事，而我们并没有为去混响准备足够的数据。
    想训练去混响时把 ``dereverb_target=True`` 打开，参考就变成干信号。
    """

    def __init__(
        self,
        clean: str | Path | list,
        noise: str | Path | list,
        rirs: str | Path | list | None = None,
        cfg: MixConfig | None = None,
        length: int = 20000,
        dereverb_target: bool = False,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or MixConfig()
        # 三个参数都既接受**目录**（递归扫描）也接受**文件列表**。
        # 支持文件列表是为 Colab 准备的：那边已经有按说话人划分好的清单，
        # 再去扫几万个文件既慢又可能把划分搞乱（训练集混进测试说话人）。
        self.clean = _as_files(clean, "干净语音")
        self.noise = _as_files(noise, "噪声")
        self.rirs = _as_files(rirs, "RIR", allow_empty=True) if rirs is not None else []
        self.length = length
        self.dereverb_target = dereverb_target
        self.seed = seed
        self.seg_len = int(self.cfg.segment_seconds * self.cfg.sample_rate)

    def __len__(self) -> int:
        return self.length

    def _load_segment(self, path: Path, rng: random.Random) -> np.ndarray:
        """读一个文件并随机截取一段。太短就循环补齐。"""
        try:
            x = read_audio(path, self.cfg.sample_rate)
        except Exception:  # noqa: BLE001
            # 数据集里混进坏文件是常态（尤其是爬来的噪声库）。
            # 静默跳过并返回静音，比让整个训练崩掉好。
            return np.zeros(self.seg_len)
        if x.size == 0:
            return np.zeros(self.seg_len)
        if x.size < self.seg_len:
            x = np.tile(x, int(np.ceil(self.seg_len / x.size)))
        start = rng.randrange(0, max(1, x.size - self.seg_len + 1))
        return x[start : start + self.seg_len]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 每个样本用独立的随机源，且种子与 idx 和 epoch 无关地散开 ——
        # 多 worker 下若共用全局 random，所有 worker 会产出相同的混音组合。
        rng = random.Random((self.seed * 1_000_003 + idx * 7919) & 0xFFFFFFFF)
        nprng = np.random.default_rng((self.seed * 31 + idx * 104729) & 0xFFFFFFFF)

        clean = self._load_segment(self.clean[rng.randrange(len(self.clean))], rng)

        # 混响
        wet = clean
        if rng.random() < self.cfg.reverb_prob:
            if self.rirs:
                rir = read_audio(self.rirs[rng.randrange(len(self.rirs))], self.cfg.sample_rate)
                if rir.size > 1:
                    wet = apply_rir(clean, rir)
            else:
                from rtse.data.synth import make_rir

                wet = apply_rir(clean, make_rir(rng.uniform(0.2, 0.8), rng=nprng))

        target = clean if self.dereverb_target else wet

        # 加噪
        if rng.random() < self.cfg.noise_prob:
            noise = self._load_segment(self.noise[rng.randrange(len(self.noise))], rng)
            snr = rng.uniform(*self.cfg.snr_range)
            noisy, _ = mix_at_snr(wet, noise, snr, rng=nprng)
        else:
            noisy = wet

        # 随机整体增益，让模型对输入音量不敏感。
        # 增益要**同时**作用于输入和参考，否则等于教模型去猜绝对音量。
        gain = 10.0 ** (rng.uniform(-6.0, 3.0) / 20.0)
        peak = max(np.max(np.abs(noisy)), np.max(np.abs(target)), 1e-9)
        # 会削波时**直接按峰值归一到 0.9**，而不是在 gain 的基础上再缩。
        # 后者（gain/peak*0.9）在 gain > 1.1 时结果仍会超过 1 —— 训练数据被削波，
        # 而模型推理时见到的是没削波的输入，两者分布对不上。
        scale = gain if peak * gain <= 0.99 else 0.9 / peak

        return (
            torch.from_numpy((noisy * scale).astype(np.float32)),
            torch.from_numpy((target * scale).astype(np.float32)),
        )


def stft_torch(x: torch.Tensor, cfg: STFTConfig = DEFAULT_CONFIG) -> torch.Tensor:
    """torch 版 STFT，返回 ``(B, 2, T, F)`` 实虚双通道。

    **必须与 numpy 版 ``rtse.audio.stft.stft`` 逐样本一致**，
    否则训练时的输入分布与推理时不同，模型一上线就掉点。
    一致性由 ``tests/test_torch_stft_parity.py`` 强制。

    对齐要点：
    - ``center=False``（numpy 版是严格因果的，不做 center padding）
    - 左侧手动补 ``n_fft - hop`` 个零，与 numpy 版的 pad 完全相同
    - 同一个归一化 sqrt-Hann 窗
    """
    win = torch.from_numpy(cfg.window).to(x.device, x.dtype)

    # 帧数**必须**用与 numpy 版同一个 num_frames()。
    # 早期这里自己推了一遍帧数，比 numpy 版少一帧，结果是尾部约 256 个样本
    # 重建不出来 —— 单看 STFT 或 iSTFT 的对拍都能过（重叠部分是一致的），
    # 只有做完整往返才暴露。见 docs/ISSUES.md I-14。
    n_fr = num_frames(x.shape[-1], cfg)
    need = (n_fr - 1) * cfg.hop + cfg.n_fft
    x = torch.nn.functional.pad(x, (cfg.pad, max(0, need - cfg.pad - x.shape[-1])))

    spec = torch.stft(
        x, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.n_fft,
        window=win, center=False, return_complex=True,
    )  # (B, F, T)
    spec = spec.transpose(-1, -2)  # (B, T, F)
    return torch.stack([spec.real, spec.imag], dim=1)


def istft_torch(spec: torch.Tensor, length: int | None = None, cfg: STFTConfig = DEFAULT_CONFIG):
    """torch 版 iSTFT：**裸重叠相加**，与 numpy 版逐样本一致。

    不用 ``torch.istft``，原因有两条（见 docs/ISSUES.md I-14）：

    1. 它会做**窗和归一化**（除以 ``sum(w^2)`` 的包络），而我们的约定是裸 OLA ——
       因为流式 iSTFT 没有能力做全局归一化，只有两边都裸 OLA 才可能一致。
    2. 它强制 **NOLA 检查**，而 sqrt-Hann 的首尾样本窗值为 0，
       ``center=False`` 下边缘帧必然触发 ``window overlap add min: 1`` 直接报错。

    这里用 ``F.fold`` 手写重叠相加，与 ``rtse.audio.stft.istft`` 是同一个算法。
    """
    win = torch.from_numpy(cfg.window).to(spec.device, spec.dtype)
    cplx = torch.complex(spec[:, 0], spec[:, 1])  # (B, T, F)
    frames = torch.fft.irfft(cplx, n=cfg.n_fft, dim=-1) * win  # (B, T, n_fft)

    n_fr = frames.shape[1]
    total = (n_fr - 1) * cfg.hop + cfg.n_fft
    y = torch.nn.functional.fold(
        frames.transpose(1, 2),  # (B, n_fft, T)
        output_size=(1, total),
        kernel_size=(1, cfg.n_fft),
        stride=(1, cfg.hop),
    ).reshape(frames.shape[0], total)

    y = y[..., cfg.pad :]
    if length is not None:
        y = y[..., :length] if y.shape[-1] >= length else torch.nn.functional.pad(
            y, (0, length - y.shape[-1])
        )
    return y


def collate_specs(batch):
    noisy = torch.stack([b[0] for b in batch])
    clean = torch.stack([b[1] for b in batch])
    return noisy, clean
