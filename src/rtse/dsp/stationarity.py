"""噪声平稳性度量：把噪声自动分成"稳态"与"非稳态"两类。

**为什么需要它**：整个项目最核心的对照——传统 DSP 在稳态噪声上够用、
在非稳态噪声上失效，神经网络反过来——要成立，前提是**能把噪声按平稳性分组**。
但 DNS Challenge 的噪声来自 AudioSet / Freesound，文件本身**不带平稳性标注**
（`isReal`/T60 那类元数据只有 RIR 有）。靠文件名猜（"fan"→稳态、"door"→非稳态）
既不可靠也不可复现——同一个 AudioSet 标签下混着差别很大的录音。所以这里直接从信号算。

判据：**去趋势后的帧能量动态范围**
------------------------------------
分三步，每一步都是为了修掉上一步暴露出来的问题（这三步是实测迭代出来的，
不是照搬公式）：

1. **时间轴平滑（5 帧）再度量**。单帧周期图本身是个高方差估计量，
   白噪声的帧间谱抖动很大——那是**估计误差**不是真实的非平稳。
   不平滑的话白噪声会被判成非稳态。这就是 Welch 平均的思路。
2. **对帧能量去趋势**（减掉约 1 秒的滑动中值），只保留**快速**起伏。
   不去趋势的话，车载噪声那种缓慢起伏（0.2 Hz 幅度调制）会拿到 16.7 dB 的
   动态范围，被误判成非稳态——但 MCRA 对这种慢变化跟得毫无压力，
   它根本不是问题噪声。**真正打垮 MCRA 的是快速突变，不是总变化范围。**
3. 用 **P95 − P05** 而不是最大最小值差，避免被单个野点带偏。

关于谱通量（保留为诊断量，**不参与判决**）
--------------------------------------------
最初的设计里还有第二个特征——相邻帧归一化谱的 L2 距离，本意是捕捉
"能量平稳但频谱形状一直在动"的 babble 型噪声。**实测证明它不好用**：
白噪声的谱通量（0.140）反而**高于** babble（0.096），因为周期图的估计方差
在归一化之后仍然主导着帧间差。而去趋势动态范围已经能把 babble 分出来
（10.4 dB vs 稳态组的 0.04~7.0 dB），谱通量再加进判决只会引入误判。

所以它被降级成一个**诊断输出**：数值仍然算出来供分析用，但
:func:`is_stationary` 不看它。与其保留一个看起来更"全面"、实际会帮倒忙的特征，
不如如实记录"试过、无效、原因是什么"。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig, stft

__all__ = ["StationarityFeatures", "stationarity_features", "is_stationary",
           "DEFAULT_DR_THRESHOLD_DB"]

_EPS = 1e-12

#: 时间轴平滑窗（帧）。5 帧 @16 ms 帧移 = 80 ms，够压掉周期图方差，
#: 又远短于我们关心的冲激事件（键盘敲击约 50 ms，但事件之间间隔上百毫秒）。
_SMOOTH_FRAMES = 5

#: 去趋势用的滑动中值窗（帧）。63 帧 ≈ 1 秒。
#: 比 MCRA 的最小值跟踪窗（2 秒）短，保证"MCRA 跟得上的慢变化"都被当成趋势去掉。
_DETREND_FRAMES = 63

#: 去趋势动态范围（dB）的分类门限。低于它算稳态。
#: **这个值是在本项目 7 类合成噪声上标定的**：
#: white 0.56 / hum 0.04 / car 3.52 / pink 6.98 都在门限下，
#: babble 10.4 / cafeteria 10.7 / keyboard 338 在门限上。
#: 两侧各有约 1.5 dB 余量——不算宽裕。真实录音的分布更宽更连续，
#: 这个门限**需要在真实噪声上重新标定**，所以它是参数不是常数。
DEFAULT_DR_THRESHOLD_DB = 9.0


@dataclass
class StationarityFeatures:
    """平稳性特征。数值本身也有用——可以按它给噪声排序分桶，不只是二分类。"""

    detrended_dynamic_range_db: float
    """去趋势帧能量的 P95 − P05（dB）。**判决依据**，越大越不平稳。"""
    spectral_flux: float
    """相邻帧归一化谱的平均 L2 距离。**仅供诊断，不参与判决**（原因见模块文档）。"""
    n_frames: int

    def as_dict(self) -> dict:
        return {
            "detrended_dynamic_range_db": round(self.detrended_dynamic_range_db, 3),
            "spectral_flux": round(self.spectral_flux, 4),
            "n_frames": self.n_frames,
        }


def stationarity_features(
    x: np.ndarray, cfg: STFTConfig = DEFAULT_CONFIG
) -> StationarityFeatures:
    """算一段音频的平稳性特征。

    信号太短（不足 ``_DETREND_FRAMES // 2`` 帧，约 0.5 秒）时去趋势没有意义，
    返回 ``n_frames`` 供调用方判断并跳过——真实噪声库里混着几百毫秒的碎片文件。
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    spec = stft(x, cfg)  # (T, F)
    n_fr = spec.shape[0]
    if n_fr < 4:
        return StationarityFeatures(0.0, 0.0, n_fr)

    power = np.abs(spec) ** 2

    # 步骤 1：时间轴平滑，压掉周期图的估计方差
    k = np.ones(_SMOOTH_FRAMES) / _SMOOTH_FRAMES
    power = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, power)

    # 步骤 2+3：帧能量去趋势后取分位数差
    energy_db = 10.0 * np.log10(power.sum(axis=1) + _EPS)
    win = min(_DETREND_FRAMES, n_fr if n_fr % 2 else n_fr - 1)
    trend = median_filter(energy_db, size=max(win, 3), mode="nearest")
    resid = energy_db - trend
    p95, p05 = np.percentile(resid, [95, 5])

    # 诊断量：谱通量（归一化后求帧间差，与整体音量变化解耦）
    mag = np.sqrt(power)
    unit = mag / np.maximum(np.linalg.norm(mag, axis=1, keepdims=True), _EPS)
    flux = float(np.mean(np.linalg.norm(np.diff(unit, axis=0), axis=1))) if n_fr > 1 else 0.0

    return StationarityFeatures(float(p95 - p05), flux, n_fr)


def is_stationary(
    x: np.ndarray,
    cfg: STFTConfig = DEFAULT_CONFIG,
    dr_threshold_db: float = DEFAULT_DR_THRESHOLD_DB,
) -> bool:
    """二分类判决：是否为稳态噪声。

    太短的信号返回 ``False``（不计入稳态组）——无法可靠判断时，
    保守地不把它放进"DSP 应该表现好"的那一组，避免稀释对照的说服力。
    """
    f = stationarity_features(x, cfg)
    if f.n_frames < _DETREND_FRAMES // 2:
        return False
    return f.detrended_dynamic_range_db < dr_threshold_db
