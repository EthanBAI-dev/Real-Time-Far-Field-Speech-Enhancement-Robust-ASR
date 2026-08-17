"""噪声平稳性判别测试。

这个模块的存在是为了给 DNS Challenge 的真实噪声自动打"稳态/非稳态"标签
（那批数据不带这个标注），而整个项目最核心的对照——DSP 在稳态噪声上够用、
在非稳态噪声上失效——完全依赖这个分组是否可靠。

所以测试的重点是**分类正确性**和**那三步预处理各自是否必要**，
而不是"函数能返回一个数"。
"""

import numpy as np
import pytest

from rtse.data.synth import make_noise
from rtse.dsp.stationarity import (
    DEFAULT_DR_THRESHOLD_DB,
    is_stationary,
    stationarity_features,
)

RNG_SEED = 7
DUR = 16000 * 10

#: 真值来自**构造方式**，不是听感：white/pink 是平稳随机过程，hum 是固定谐波；
#: babble/cafeteria 谱形状随音节变化，keyboard 是冲激序列。
#:
#: ⚠️ **`car` 刻意不在这张表里**。它是 1/f^2.5 的陡峭低频加 0.2 Hz 慢调制，
#: 实测 12 个随机种子的动态范围在 0.67~16.22 dB 之间，**横跨门限**，
#: 判决随随机实现翻转。最初这里断言它是稳态，用固定种子**碰巧过了**——
#: 那是一条靠运气通过的测试，比没有测试更糟。现在改成如实记录它是边界情况
#: （见 test_car_like_noise_is_a_documented_borderline_case）。
EXPECTED = {
    "white": True,
    "pink": True,
    "hum": True,
    "babble": False,
    "cafeteria": False,
    "keyboard": False,
}


def _noise(kind: str) -> np.ndarray:
    return make_noise(kind, DUR, np.random.default_rng(RNG_SEED))


@pytest.mark.parametrize("kind,expected", sorted(EXPECTED.items()))
def test_classifies_known_noise_types(kind, expected):
    assert is_stationary(_noise(kind)) is expected, (
        f"{kind} 判错了，特征={stationarity_features(_noise(kind)).as_dict()}"
    )


def test_classification_is_stable_across_random_seeds():
    """**跨随机种子稳定**——这比单个种子上分类正确重要得多。

    合成噪声每次调用都是新的随机实现。如果判决会随实现翻转，
    那"分类正确"就只是运气好，换一批数据就会翻车。
    这条测试对每类跑多个种子，要求判决**全部一致**。

    （`car` 不在 EXPECTED 里，正是因为它过不了这一关——见下一条测试。）
    """
    for kind, expected in EXPECTED.items():
        got = [is_stationary(make_noise(kind, DUR, np.random.default_rng(s)))
               for s in range(8)]
        assert all(g is expected for g in got), (
            f"{kind} 判决随种子翻转: {got}，说明它其实是边界情况，"
            f"不该出现在 EXPECTED 里"
        )


def test_threshold_has_margin_on_both_sides():
    """门限两侧都要有余量，不能刚好卡在某一类的数值上。

    余量太小意味着换一批噪声就会翻车。这条测试把当前余量固定下来，
    以后调整预处理若把某一类推到门限附近，会立刻暴露。
    """
    dr = {k: stationarity_features(_noise(k)).detrended_dynamic_range_db for k in EXPECTED}
    stationary_max = max(dr[k] for k, v in EXPECTED.items() if v)
    nonstationary_min = min(dr[k] for k, v in EXPECTED.items() if not v)
    assert stationary_max < DEFAULT_DR_THRESHOLD_DB < nonstationary_min
    assert DEFAULT_DR_THRESHOLD_DB - stationary_max > 0.5, f"稳态侧余量不足: {dr}"
    assert nonstationary_min - DEFAULT_DR_THRESHOLD_DB > 0.5, f"非稳态侧余量不足: {dr}"


def test_car_like_noise_is_a_documented_borderline_case():
    """如实记录：极低频主导 + 慢调制的噪声，本模块**分不稳**。

    这不是"待修的 bug"，是已知且已接受的限制（模块文档里有完整说明，
    包括试过限带规避、反而更糟这条否定结果）。
    写成测试是为了让这个限制**可见且可量化**——哪天预处理改动让 car 稳定了，
    这条测试会失败，提醒把它移回 EXPECTED。
    """
    dr = np.array([stationarity_features(
        make_noise("car", DUR, np.random.default_rng(s))).detrended_dynamic_range_db
        for s in range(12)])
    assert dr.min() < DEFAULT_DR_THRESHOLD_DB < dr.max(), (
        f"car 不再横跨门限了（范围 {dr.min():.2f}~{dr.max():.2f}）——"
        f"如果这是预处理改进带来的，把 car 移回 EXPECTED 并更新模块文档"
    )


def test_detrending_actually_reduces_dynamic_range():
    """**去趋势这一步是必要的**，不是可有可无的精细化。

    验证方式不依赖某一类噪声的具体判决（那会踩 car 那个坑）：
    构造一个"平稳白噪声 + 缓慢增益漂移"的信号——它在 MCRA 看来完全跟得上，
    应当算稳态。不去趋势的话那段漂移会被算进动态范围，判成非稳态。
    """
    rng = np.random.default_rng(3)
    n = 16000 * 20
    t = np.arange(n) / 16000
    drift = 10.0 ** (6.0 * np.sin(2 * np.pi * 0.05 * t) / 20.0)  # ±6 dB，0.05 Hz
    x = rng.standard_normal(n) * drift
    assert is_stationary(x), (
        "缓慢增益漂移的白噪声被判成非稳态，说明去趋势没起作用——"
        f"DR={stationarity_features(x).detrended_dynamic_range_db:.2f} dB"
    )


def test_spectral_flux_is_reported_but_would_misclassify_if_used():
    """记录一个**否定结果**：谱通量单独用会判错，所以它不参与判决。

    最初设计里谱通量是第二个判据（本意是抓"能量平稳但频谱在动"的 babble）。
    实测发现白噪声的谱通量反而高于 babble——周期图的估计方差在归一化之后
    依然主导帧间差。这条测试把这个事实钉死：如果以后有人想把 flux 加回判决，
    它会提醒对方这条路已经试过且走不通。
    """
    white_flux = stationarity_features(_noise("white")).spectral_flux
    babble_flux = stationarity_features(_noise("babble")).spectral_flux
    assert white_flux > babble_flux, (
        "白噪声的谱通量不再高于 babble——如果预处理改动让这个关系反转了，"
        "可以重新评估把 flux 纳入判决；在此之前它只是诊断量。"
    )


def test_too_short_signal_is_not_counted_as_stationary():
    """短到无法可靠判断时返回 False（不计入稳态组）。

    真实噪声库里混着几百毫秒的碎片文件。保守处理的理由是：
    把不确定的样本放进"DSP 应该表现好"那一组，会稀释整个对照的说服力。
    """
    assert is_stationary(np.random.default_rng(0).standard_normal(1600)) is False
    assert stationarity_features(np.zeros(100)).n_frames < 4


def test_features_are_finite_on_silence_and_dc():
    """退化输入不能产生 nan/inf——真实数据里一定会有全零段和直流偏置。"""
    for x in (np.zeros(DUR), np.ones(DUR)):
        f = stationarity_features(x)
        assert np.isfinite(f.detrended_dynamic_range_db)
        assert np.isfinite(f.spectral_flux)
