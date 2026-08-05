"""STFT 地基测试。这几条不过，后面所有指标都不用看了。"""

import numpy as np
import pytest

from rtse.audio.stft import (
    STFTConfig,
    StreamingISTFT,
    StreamingSTFT,
    check_cola,
    istft,
    num_frames,
    sqrt_hann,
    stft,
)

RNG = np.random.default_rng(20260805)


def _signals():
    """一组有代表性的测试信号：随机噪声、纯音、扫频、脉冲、静音、非整数帧长。"""
    sr = 16000
    t = np.arange(sr) / sr
    return {
        "noise": RNG.standard_normal(16000),
        "tone": np.sin(2 * np.pi * 440 * t),
        "chirp": np.sin(2 * np.pi * (100 + 3000 * t) * t),
        "impulse": np.eye(1, 16000, 3000).ravel(),
        "silence": np.zeros(16000),
        "ragged": RNG.standard_normal(12345),  # 非 hop 整数倍，专门打边界
        "tiny": RNG.standard_normal(100),  # 比一个窗还短
    }


@pytest.mark.parametrize("hop_div", [2, 4, 8])
def test_cola_holds_after_normalization(hop_div):
    """归一化后的 sqrt-Hann 在 50% / 75% / 87.5% 重叠下都应满足 COLA。"""
    cfg = STFTConfig(n_fft=512, hop=512 // hop_div)
    assert check_cola(cfg.n_fft, cfg.hop) < 1e-12


@pytest.mark.parametrize("hop_div,expected_gain", [(2, 1.0), (4, 2.0), (8, 4.0)])
def test_raw_sqrt_hann_gain_is_only_unity_at_50pct(hop_div, expected_gain):
    """未归一化的 sqrt-Hann 只在 50% 重叠时增益为 1，其余会整体放大。

    这是 I-06 的回归测试：把"sqrt-Hann 天然满足 COLA"这个常见误解钉死。
    """
    n_fft, hop = 512, 512 // hop_div
    raw = sqrt_hann(n_fft)  # 不传 hop == 不归一化
    prod = raw * raw
    gain = sum(prod[k * hop : (k + 1) * hop] for k in range(n_fft // hop))
    assert np.allclose(gain, expected_gain, atol=1e-12)


@pytest.mark.parametrize("hop_div", [2, 4])
def test_perfect_reconstruction_at_other_hops(hop_div):
    """归一化生效的最终判据：换 hop 后依然完美重构（而不只是 COLA 数值好看）。"""
    cfg = STFTConfig(n_fft=512, hop=512 // hop_div)
    x = RNG.standard_normal(9000)
    y = istft(stft(x, cfg), length=x.size, cfg=cfg)
    assert np.max(np.abs(y - x)) < 1e-10


def test_symmetric_hann_would_fail_cola():
    """反证：对称 Hann（np.hanning）不满足 COLA。

    这条测试存在的意义是把坑钉死 —— 误差约 1e-3，小到跑起来"听着没问题"，
    大到足以污染 SI-SDR 指标。
    """
    n_fft, hop = 512, 256
    bad = np.sqrt(np.hanning(n_fft))
    prod = bad * bad
    acc = sum(prod[k * hop : (k + 1) * hop] for k in range(n_fft // hop))
    assert np.max(np.abs(acc - 1.0)) > 1e-4


@pytest.mark.parametrize("name", list(_signals()))
def test_perfect_reconstruction(name):
    """完美重构：不做任何处理时 istft(stft(x)) 应逐样本等于 x。"""
    x = _signals()[name]
    y = istft(stft(x), length=x.size)
    assert y.shape == x.shape
    err = np.max(np.abs(y - x))
    assert err < 1e-10, f"{name}: 重构误差 {err:.3e} 超标"


def test_num_frames_covers_all_samples():
    """帧数必须足够让最后一个样本也落在 COLA 稳态区内。"""
    cfg = STFTConfig()
    for n in [1, 255, 256, 257, 511, 512, 513, 16000, 12345]:
        n_fr = num_frames(n, cfg)
        # 最后一帧的起点 + 窗长，必须覆盖到填充坐标下的最后一个有效样本
        assert (n_fr - 1) * cfg.hop >= cfg.pad + n - 1 - (cfg.n_fft - 1)
        assert (n_fr - 1) * cfg.hop + cfg.n_fft >= cfg.pad + n


def test_streaming_stft_matches_offline():
    """流式分析必须与整段分析逐帧完全一致。"""
    cfg = STFTConfig()
    x = RNG.standard_normal(16000)
    offline = stft(x, cfg)

    n_fr = offline.shape[0]
    padded = np.zeros((n_fr - 1) * cfg.hop + cfg.n_fft)
    padded[cfg.pad : cfg.pad + x.size] = x
    # 流式缓冲初始化为 n_fft 个零，等价于左侧已补 pad；因此从 pad 之后开始喂
    stream_in = padded[cfg.pad :]

    sa = StreamingSTFT(cfg)
    got = np.stack([sa.push(stream_in[i * cfg.hop : (i + 1) * cfg.hop]) for i in range(n_fr)])

    assert got.shape == offline.shape
    assert np.max(np.abs(got - offline)) < 1e-12


def test_streaming_roundtrip_matches_offline():
    """流式 分析→合成 全链路必须与离线逐样本一致，且延迟恰为 pad 个样本。"""
    cfg = STFTConfig()
    x = RNG.standard_normal(8000)
    n_fr = num_frames(x.size, cfg)

    feed = np.zeros(n_fr * cfg.hop)
    feed[: x.size] = x  # 流式侧直接喂原始信号，补零由内部缓冲承担

    sa, sy = StreamingSTFT(cfg), StreamingISTFT(cfg)
    out = np.concatenate(
        [sy.push(sa.push(feed[i * cfg.hop : (i + 1) * cfg.hop])) for i in range(n_fr)]
    )

    # 流式输出处在"填充坐标系"，前 pad 个样本对应左侧补零区，丢弃后即为原信号
    recon = out[cfg.pad : cfg.pad + x.size]
    assert recon.size == x.size
    err = np.max(np.abs(recon - x))
    assert err < 1e-10, f"流式往返误差 {err:.3e}"


def test_streaming_reset_is_clean():
    """reset() 后必须回到初始状态，否则 Web 端切换方法时会串音。"""
    cfg = STFTConfig()
    sa = StreamingSTFT(cfg)
    block = RNG.standard_normal(cfg.hop)
    first = sa.push(block)
    sa.push(RNG.standard_normal(cfg.hop))
    sa.reset()
    assert np.max(np.abs(sa.push(block) - first)) < 1e-15


def test_spectrum_shape_and_dtype():
    cfg = STFTConfig()
    spec = stft(RNG.standard_normal(16000), cfg)
    assert spec.shape[1] == cfg.n_freq == 257
    assert spec.dtype == np.complex128


def test_magnitude_db_is_calibrated_to_dbfs():
    """满幅正弦必须精确读到 0 dBFS。

    这条锁死频谱 dB 的标定（I-12 的回归测试）。它一旦漂移，
    前端所有色标范围、评测报告里的所有 dB 数字都会跟着错，
    而且错得很隐蔽 —— 频谱图只是"看起来偏亮/偏暗"，不会报错。
    """
    from rtse.audio.stft import magnitude_db, spec_ref

    cfg = STFTConfig()
    t = np.arange(SR := 16000) / SR
    sine = np.sin(2 * np.pi * 1000 * t)
    peak_db = float(magnitude_db(stft(sine, cfg), cfg).max())
    assert abs(peak_db) < 0.05, f"满幅正弦读到 {peak_db:.3f} dBFS，应为 0"

    # 幅度减半应精确对应 -6.02 dB
    half_db = float(magnitude_db(stft(0.5 * sine, cfg), cfg).max())
    assert abs((peak_db - half_db) - 6.0206) < 0.01

    assert spec_ref(cfg) > 1.0  # 参考值确实远大于 1，这正是必须归一化的原因


def test_rejects_bad_config():
    """n_fft 不是 hop 整数倍时必须直接拒绝，而不是悄悄产生错误结果。"""
    with pytest.raises(ValueError, match="整数倍"):
        STFTConfig(n_fft=512, hop=300)


def test_rejects_wrong_block_size():
    cfg = STFTConfig()
    with pytest.raises(ValueError, match="hop"):
        StreamingSTFT(cfg).push(np.zeros(cfg.hop + 1))
