"""音频读写与重采样。

统一约定：项目内部一律使用 **16 kHz、单声道、float64、幅度归一到 [-1, 1]** 的
一维 numpy 数组。所有外部音频在入口处就转换到这个形式，绝不让采样率或声道数的
差异渗透到算法层 —— 这类问题一旦漏进去，表现是"某些文件指标特别差"，
排查成本远高于在入口处统一。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from rtse import SAMPLE_RATE

__all__ = ["read_audio", "write_audio", "resample", "to_mono", "peak_normalize", "rms", "db"]


def to_mono(x: np.ndarray) -> np.ndarray:
    """多声道取平均降为单声道。"""
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x.reshape(-1)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """重采样。

    用 soxr 而不是 ``scipy.signal.resample_poly``：soxr 的抗混叠滤波器质量更高，
    且不依赖 numba（librosa 那条链路在 Windows 上是常见的安装麻烦来源）。
    """
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float64)
    return soxr.resample(np.asarray(x, dtype=np.float64), sr_in, sr_out, quality="VHQ")


def read_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """读取任意音频文件为 16 kHz 单声道 float64。

    读取顺序刻意是 **先降声道、再重采样**：反过来会对多余声道做无用的重采样计算。
    """
    data, sr_in = sf.read(str(path), dtype="float64", always_2d=False)
    return resample(to_mono(data), sr_in, sr)


def write_audio(
    path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE, guard_clipping: bool = True
) -> float:
    """写 16-bit PCM WAV，返回实际施加的增益（1.0 表示未缩放）。

    削波保护：增强算法（尤其是维纳滤波的低 SNR 段）偶尔会产生 |x| > 1 的样本，
    直接写 16-bit 会绕回成刺耳的爆音，听感上像算法崩了，实际只是溢出。
    这里等比缩放而不是硬截断，保留波形形状。

    ⚠️ **重缩放会改变绝对幅度**，因此：
    - 指标一律在**内存中的原始数组**上计算，绝不"写盘再读回来算"
      （否则 SDR、SegSNR 这类非尺度不变的指标会被这个增益污染，见 ISSUES.md I-07）；
    - 返回增益值，调用方需要精确幅度时可据此还原；
    - 评测流程如需绕过，传 ``guard_clipping=False``。
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    gain = 1.0
    if guard_clipping and x.size:
        peak = float(np.max(np.abs(x)))
        if peak > 1.0:
            gain = 1.0 / peak
            x = x * gain
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_16")
    return gain


def peak_normalize(x: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(x))
    return x * (target / peak) if peak > 0 else x


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def db(x: float, floor: float = 1e-12) -> float:
    return 20.0 * np.log10(max(float(x), floor))
