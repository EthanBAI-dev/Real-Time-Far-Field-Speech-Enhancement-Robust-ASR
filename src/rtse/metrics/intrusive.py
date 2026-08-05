"""有参考指标：需要干净参考信号。

SI-SDR 自行实现（公式简单且必须完全可控），STOI/ESTOI/PESQ 走成熟的第三方实现
（心理声学模型自己写容易出微妙偏差，而这类指标的价值恰恰在于可比性）。
"""

from __future__ import annotations

import numpy as np

from rtse import SAMPLE_RATE

__all__ = ["si_sdr", "sdr", "seg_snr", "stoi", "estoi", "pesq", "align_lengths"]

_EPS = 1e-10


def align_lengths(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把两路信号裁到相同长度。

    增强算法（尤其是流式管线）常会因为帧对齐差出几十个样本。
    差几个样本对指标影响很小，但长度不等会直接抛异常，所以统一在这里截齐。
    差异超过一帧（16 ms）时说明是真的对齐 bug，抛错而不是默默截断。
    """
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    n = min(ref.size, est.size)
    if abs(ref.size - est.size) > SAMPLE_RATE * 0.016:
        raise ValueError(
            f"两路信号长度相差 {abs(ref.size - est.size)} 样本（>1 帧），"
            "这不是正常的帧对齐误差，请检查管线"
        )
    return ref[:n], est[:n]


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-Invariant SDR，单位 dB。

    先把参考信号投影到估计信号上求最优缩放，再算"目标 / 失真"能量比：

        s_target = <est, ref> / ||ref||^2 * ref
        e_noise  = est - s_target
        SI-SDR   = 10 log10(||s_target||^2 / ||e_noise||^2)

    对整体增益不敏感是它相对普通 SDR 的关键优势：语音增强常伴随音量变化，
    普通 SDR 会把"音量小了 3 dB"误判成失真。
    """
    ref, est = align_lengths(ref, est)
    # 去均值：直流分量会污染投影
    ref = ref - ref.mean()
    est = est - est.mean()

    ref_energy = np.sum(ref**2) + _EPS
    scale = np.sum(est * ref) / ref_energy
    target = scale * ref
    noise = est - target
    return float(10.0 * np.log10((np.sum(target**2) + _EPS) / (np.sum(noise**2) + _EPS)))


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """普通 SDR（不做尺度不变），与 SI-SDR 并列给出以显示增益变化的影响。"""
    ref, est = align_lengths(ref, est)
    noise = est - ref
    return float(10.0 * np.log10((np.sum(ref**2) + _EPS) / (np.sum(noise**2) + _EPS)))


def seg_snr(
    ref: np.ndarray,
    est: np.ndarray,
    frame_len: int = 512,
    hop: int = 256,
    lo: float = -10.0,
    hi: float = 35.0,
) -> float:
    """分段信噪比（dB），只统计有语音的帧。

    截断到 [-10, 35] dB 是标准做法：静音帧的 SNR 会趋向 ±inf，
    不截断的话整个指标会被几帧静音主导，失去意义。
    """
    ref, est = align_lengths(ref, est)
    n_fr = max(1, (ref.size - frame_len) // hop + 1)
    vals = []
    # 用全局能量的 -40 dB 作为"这一帧算不算语音"的门限
    thresh = np.mean(ref**2) * 1e-4
    for i in range(n_fr):
        s = i * hop
        r = ref[s : s + frame_len]
        e = est[s : s + frame_len]
        r_energy = np.sum(r**2)
        if r_energy < thresh:
            continue
        d = np.sum((r - e) ** 2) + _EPS
        vals.append(np.clip(10.0 * np.log10(r_energy / d), lo, hi))
    return float(np.mean(vals)) if vals else float("nan")


def stoi(ref: np.ndarray, est: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """短时客观可懂度，取值 0~1。与 ASR 字错率相关性通常好于 PESQ。"""
    from pystoi import stoi as _stoi

    ref, est = align_lengths(ref, est)
    return float(_stoi(ref, est, sr, extended=False))


def estoi(ref: np.ndarray, est: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """扩展 STOI，对高度调制的噪声（如 babble）比 STOI 更可靠。"""
    from pystoi import stoi as _stoi

    ref, est = align_lengths(ref, est)
    return float(_stoi(ref, est, sr, extended=True))


def pesq(ref: np.ndarray, est: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    """PESQ (ITU-T P.862.2 宽带)，取值 -0.5~4.5。

    **本机不可用**：`pesq` 包只发源码分发，Windows 上需要 MSVC 现编，
    见 docs/ISSUES.md I-05。这里返回 None，评测表标 n/a。
    在 Colab（Linux）上该依赖可正常安装，PESQ 由那一侧补齐。
    """
    try:
        from pesq import pesq as _pesq
    except ImportError:
        return None
    ref, est = align_lengths(ref, est)
    try:
        return float(_pesq(sr, ref, est, "wb"))
    except Exception:
        # PESQ 对全静音、过短片段会直接抛错，这类样本返回 None 而不是让评测中断
        return None
