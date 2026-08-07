"""从房间冲激响应估计混响时间 T60（Schroeder 反向积分法）。

**为什么需要它**：要给测试集加一组"真实 RIR 分层"，就必须按 T60 把真实 RIR
分桶（否则没法做 T60 扫描这一维的受控对比，见 docs/FINDINGS.md F-07）。
真实 RIR 文件本身不带 T60 标注——DNS Challenge 有一张 `RIR_table_simple.csv`
提供标注，但它不在仓库的代码分支里，是否随数据分片一起分发无法在本机确认
（本机磁盘装不下 6GB 分片来验证）。所以这里自己实现估计，**不依赖那张表是否存在**；
表若存在，可以拿它来交叉验证本模块的准确度。

**方法**（ISO 3382 的标准做法）：

1. 能量衰减曲线（EDC，Schroeder 1965）：``EDC(t) = ∫_t^∞ h²(τ) dτ``，
   即从 t 到结尾的剩余能量。**用反向积分而不是直接看 h² 的包络**，
   因为单次冲激响应的瞬时能量起伏极大，直接拟合噪声太大；
   反向积分等价于对无穷多次测量取系综平均，曲线变得光滑可拟合。
2. 转 dB 并归一化到 0 dB 起点。
3. 在 **−5 到 −35 dB** 区间做线性拟合（即 T30），再把斜率外推到 −60 dB。
   不从 0 dB 起拟合：起点附近被直达声主导，不属于混响衰减段；
   也不拟合到 −60 dB：那里通常已经埋进本底噪声，斜率会被压平导致高估 T60。
"""

from __future__ import annotations

import numpy as np

from rtse import SAMPLE_RATE

__all__ = ["energy_decay_curve", "estimate_t60"]


def energy_decay_curve(rir: np.ndarray) -> np.ndarray:
    """Schroeder 反向积分能量衰减曲线，归一化到 0 dB 起点，返回 dB 值。"""
    h = np.asarray(rir, dtype=np.float64).reshape(-1)
    # 从直达声（最大峰）开始，丢掉之前的传播延迟——那段是静音，
    # 会被算成"衰减还没开始"，把拟合区间整体右移。
    start = int(np.argmax(np.abs(h)))
    h = h[start:]
    energy = h**2
    edc = np.cumsum(energy[::-1])[::-1]  # 反向累加 = 从 t 到结尾的剩余能量
    total = edc[0] if edc.size else 0.0
    if total <= 0:
        return np.full(h.size, -np.inf)
    return 10.0 * np.log10(np.maximum(edc / total, 1e-20))


def estimate_t60(
    rir: np.ndarray,
    sr: int = SAMPLE_RATE,
    lo_db: float = -5.0,
    hi_db: float = -35.0,
) -> float:
    """估计 T60（秒）。信号太短或衰减不足以覆盖拟合区间时返回 ``nan``。

    Args:
        rir: 一维冲激响应。
        lo_db/hi_db: 拟合区间，默认 −5~−35 dB（即 T30 外推），ISO 3382 的常用取值。
            真实 RIR 若本底噪声较高、衰减不到 −35 dB，可以放宽到 −25 dB（即 T20）。

    返回 ``nan`` 而不是抛异常或猜一个值：真实 RIR 库里混着各种质量的文件，
    调用方需要能识别"这条估不出来"并跳过，而不是拿到一个看似合理的假数字。
    """
    edc = energy_decay_curve(rir)
    if edc.size < 2 or not np.isfinite(edc[0]):
        return float("nan")

    # 找到首次跌破 lo_db / hi_db 的位置。EDC 单调不增，所以 searchsorted
    # 用在反转后的序列上即可，但直接 argmax 更直观且不依赖严格单调。
    below_lo = np.flatnonzero(edc <= lo_db)
    below_hi = np.flatnonzero(edc <= hi_db)
    if below_lo.size == 0 or below_hi.size == 0:
        return float("nan")  # 衰减深度不够，覆盖不了拟合区间
    i0, i1 = int(below_lo[0]), int(below_hi[0])
    if i1 - i0 < 2:
        return float("nan")

    t = np.arange(i0, i1) / sr
    y = edc[i0:i1]
    slope = np.polyfit(t, y, 1)[0]  # dB/秒，应为负
    if slope >= -1e-9:
        return float("nan")
    return float(-60.0 / slope)
