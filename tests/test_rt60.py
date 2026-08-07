"""T60 估计与 RIR 合成的一致性测试。

这组测试的直接由来是一个真实的 bug（docs/ISSUES.md I-22）：``make_rir()`` 曾经
把镜像阶数写死成 12，导致镜像源最远只到 0.56 秒，**目标 T60 超过约 0.6 秒之后
完全失效**——标称 0.6/0.9/1.0 的 RIR 实际测出来全是 0.58~0.63 秒。
测试集的 T60 扫描因此有两档实际上是同一个混响强度，
而这个问题在指标表里表现为"T60=0.9 反而比 0.6 好"，看起来像实验发现，
实际是数据本身错了。

所以这里的核心测试是**单调性**和**相对精度**，而不是"函数能跑通"。
"""

import numpy as np
import pytest

from rtse.data.synth import make_rir
from rtse.dsp.rt60 import energy_decay_curve, estimate_t60

RNG = np.random.default_rng(20260807)


def test_edc_is_monotonically_decreasing():
    """能量衰减曲线必须单调不增——它是"从 t 到结尾的剩余能量"，
    定义上不可能随时间增加。这条不成立说明反向积分写反了。
    """
    edc = energy_decay_curve(make_rir(0.4, rng=RNG))
    assert np.all(np.diff(edc) <= 1e-9)
    assert edc[0] == pytest.approx(0.0, abs=1e-9), "EDC 应归一化到 0 dB 起点"


def test_estimate_t60_on_ideal_exponential_decay():
    """对**解析可算**的理想指数衰减，估计值必须准确。

    这是估计器本身的正确性检验，不掺 make_rir 的近似误差：
    构造 h(t) = noise * exp(-k*t)，其中 k 由目标 T60 精确给定
    （振幅衰减 60 dB 对应 amplitude ratio 1e-3）。
    """
    sr, target = 16000, 0.5
    n = int(sr * target * 2)
    t = np.arange(n) / sr
    decay = 10.0 ** (-3.0 * t / target)  # 振幅在 t=target 时降到 1e-3，即 -60 dB
    h = RNG.standard_normal(n) * decay
    assert estimate_t60(h, sr) == pytest.approx(target, rel=0.1)


@pytest.mark.parametrize("t60", [0.3, 0.6, 0.9])
def test_make_rir_reaches_requested_t60(t60):
    """合成 RIR 的实际 T60 必须落在目标附近。

    容差放到 ±25%：Sabine 公式在高吸声系数下本身就是近似，
    再加上 make_rir 里那层高频吸收包络（自身等效 T60 约 3.45 秒）会叠加衰减。
    这个容差不是为了让测试好过——**出 bug 时的偏差是 30%~40% 且方向一致
    （长混响被压到 0.63 秒封顶），足以被这个阈值抓住**。
    """
    measured = estimate_t60(make_rir(t60, rng=RNG))
    assert measured == pytest.approx(t60, rel=0.25), f"标称 {t60}s 实测 {measured:.3f}s"


def test_longer_t60_actually_produces_longer_reverb():
    """**这条是 I-22 的回归测试**：更大的目标 T60 必须真的产生更长的混响。

    旧实现（max_order 写死 12）在这里会失败：0.6 / 0.9 / 1.2 测出来分别是
    0.58 / 0.63 / 0.63 秒，后两档几乎相同——"加大混响"这个操作在
    0.6 秒以上完全失效，而调用方拿到的却是一个看起来正常的 RIR。
    """
    measured = [estimate_t60(make_rir(t, rng=RNG)) for t in (0.3, 0.6, 0.9, 1.2)]
    assert all(np.isfinite(measured)), f"存在估不出来的 RIR: {measured}"
    for a, b in zip(measured, measured[1:]):
        assert b > a * 1.15, f"T60 没有随目标显著增长: {measured}"


def test_rir_content_extends_over_the_full_decay():
    """RIR 里必须真的有覆盖到衰减尾部的反射声，不能提前变成一片零。

    旧实现的失效方式正是这个：镜像阶数不够，0.56 秒之后全是 0，
    但数组长度按 t60 分配，于是后半段是纯零填充——
    EDC 会直接掉到 -inf，"混响"其实早就结束了。
    """
    t60 = 0.9
    rir = make_rir(t60, rng=RNG)
    nonzero = np.flatnonzero(np.abs(rir) > 1e-12)
    coverage = (nonzero[-1] + 1) / 16000
    assert coverage > t60 * 0.8, f"RIR 有效内容只到 {coverage:.3f}s，目标 T60 是 {t60}s"


def test_estimate_returns_nan_when_decay_is_insufficient():
    """衰减深度不够时返回 nan，而不是硬凑一个数字。

    真实 RIR 库里混着各种质量的文件，调用方需要能识别"这条估不出来"
    并跳过；返回一个看似合理的假值会让坏数据静默混进分桶结果。
    """
    assert np.isnan(estimate_t60(np.ones(1000)))  # 完全不衰减
    assert np.isnan(estimate_t60(np.array([1.0])))  # 太短
