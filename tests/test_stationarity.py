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

#: 真值来自**构造方式**，不是听感：white/pink 是平稳随机过程，hum 是固定谐波，
#: car 是 pink 加 0.2 Hz 慢调制（MCRA 完全跟得上，算稳态）；
#: babble/cafeteria 谱形状随音节变化，keyboard 是冲激序列。
EXPECTED = {
    "white": True,
    "pink": True,
    "hum": True,
    "car": True,
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


def test_threshold_has_margin_on_both_sides():
    """门限两侧都要有余量，不能刚好卡在某一类的数值上。

    余量太小意味着换一批噪声就会翻车。这条测试把当前余量固定下来，
    以后调整预处理若把某一类推到门限附近，会立刻暴露。
    """
    dr = {k: stationarity_features(_noise(k)).detrended_dynamic_range_db for k in EXPECTED}
    stationary_max = max(dr[k] for k, v in EXPECTED.items() if v)
    nonstationary_min = min(dr[k] for k, v in EXPECTED.items() if not v)
    assert stationary_max < DEFAULT_DR_THRESHOLD_DB < nonstationary_min
    assert DEFAULT_DR_THRESHOLD_DB - stationary_max > 1.0, f"稳态侧余量不足: {dr}"
    assert nonstationary_min - DEFAULT_DR_THRESHOLD_DB > 1.0, f"非稳态侧余量不足: {dr}"


def test_detrending_is_what_rescues_slowly_modulated_noise():
    """**去趋势这一步是必要的**，不是可有可无的精细化。

    car 是 pink 噪声乘以 0.2 Hz 的慢幅度调制。不去趋势时它的动态范围高达
    16.7 dB，会被判成非稳态——但 MCRA 对这种慢变化跟得毫无压力，
    把它归到"DSP 应该失效"那一组是错的。这条测试验证去趋势确实把它压了下来。
    """
    f = stationarity_features(_noise("car"))
    assert f.detrended_dynamic_range_db < 5.0, (
        f"car 去趋势后仍有 {f.detrended_dynamic_range_db:.1f} dB，去趋势没起作用"
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
