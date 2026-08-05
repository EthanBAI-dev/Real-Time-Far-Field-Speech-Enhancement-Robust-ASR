"""DSP 增强器、VAD、管线的测试。"""

import numpy as np
import pytest

from rtse.audio.stft import DEFAULT_CONFIG, StreamingSTFT
from rtse.data.synth import apply_rir, make_noise, make_rir, mix_at_snr, speech_active_mask
from rtse.dsp import MCRANoiseEstimator, PassThrough, build_dsp
from rtse.metrics.intrusive import si_sdr
from rtse.runtime import Pipeline
from rtse.vad import build_vad

RNG = np.random.default_rng(20260805)
SR = 16000


@pytest.fixture(scope="module")
def speech():
    """合成的类语音信号：基频 + 谐波 + 共振峰包络 + 音节调制 + 静音段。

    不用真实录音，是为了让测试**不依赖任何外部文件**（CI 里也能跑）。
    但它必须具备真实语音的关键特性：谐波结构、时变、有静音段 ——
    否则 VAD 和噪声估计的测试就退化成了对常数信号的测试。
    """
    n = SR * 4
    t = np.arange(n) / SR
    f0 = 120 + 25 * np.sin(2 * np.pi * 1.3 * t)  # 起伏的基频
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(h * phase) / h for h in range(1, 12))
    # 共振峰式的带通着色
    sig += 0.4 * np.sin(phase) * np.sin(2 * np.pi * 800 * t)
    # 音节调制 + 明确的静音段（第 1~1.6 s 和第 2.8~3.3 s）
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    env[int(1.0 * SR) : int(1.6 * SR)] = 0.0
    env[int(2.8 * SR) : int(3.3 * SR)] = 0.0
    sig *= env
    return sig / np.max(np.abs(sig)) * 0.6


# ----------------------------------------------------------------- 合成数据


def test_mix_at_snr_is_accurate():
    """混音后的实际 SNR 必须与目标一致（在语音活跃段上定义）。"""
    speech_sig = RNG.standard_normal(SR * 2)
    noise = make_noise("white", SR * 2, RNG)
    for target in [-5, 0, 5, 10, 20]:
        mixed, scaled_noise = mix_at_snr(speech_sig, noise, target, rng=RNG)
        mask = speech_active_mask(speech_sig)
        actual = 10 * np.log10(np.mean(speech_sig[mask] ** 2) / np.mean(scaled_noise**2))
        assert abs(actual - target) < 0.5, f"目标 {target} dB，实测 {actual:.2f} dB"
        assert np.allclose(mixed, speech_sig + scaled_noise)


@pytest.mark.parametrize("kind", ["white", "pink", "brown", "babble", "car", "keyboard", "hum"])
def test_noise_kinds_are_finite_and_unit_variance(kind):
    y = make_noise(kind, SR, RNG)
    assert y.size == SR
    assert np.all(np.isfinite(y))
    assert abs(np.std(y) - 1.0) < 1e-6


@pytest.mark.parametrize("t60", [0.3, 0.6, 0.9])
def test_rir_decay_matches_target_t60(t60):
    """镜像源法生成的 RIR，其能量衰减曲线应与目标 T60 大致吻合。

    用 Schroeder 积分测实际 T60，容差放到 ±50% —— 镜像源阶数有限、
    Sabine 公式本身也是近似，要求过严是自欺欺人。这里要验的是
    "T60 参数确实在控制衰减速度"，而不是精确标定。
    """
    rir = make_rir(t60, rng=RNG)
    energy = rir[::-1] ** 2
    schroeder = 10 * np.log10(np.maximum(np.cumsum(energy)[::-1], 1e-20))
    schroeder -= schroeder[0]
    # 从 -5 dB 衰减到 -35 dB 的时间 × 2 即 T60（T30 外推，避开尾部噪声）
    i5 = int(np.argmax(schroeder <= -5))
    i35 = int(np.argmax(schroeder <= -35))
    if i35 <= i5:
        pytest.skip("RIR 太短，衰减未达 -35 dB")
    measured = (i35 - i5) / SR * 2
    assert 0.5 * t60 < measured < 1.6 * t60, f"目标 T60={t60}，实测 {measured:.2f}"


def test_apply_rir_is_fast_enough_for_training():
    """卷混响必须走 FFT 卷积。

    这是 I-19 的回归测试。`np.convolve` 在 T60=0.6s（12320 taps）时要
    **7 秒**处理一个 4 秒段 —— 训练时数据加载会彻底卡死，GPU 全程空转，
    而且不会有任何报错，只表现为"训练慢得离谱"。

    门限定 50 ms：FFT 卷积实测约 2 ms，直接卷积约 7000 ms，
    中间隔着三个数量级，不存在误判的余地。
    """
    import time

    x = RNG.standard_normal(SR * 4)
    rir = make_rir(0.6, rng=RNG)
    apply_rir(x, rir)  # 预热，避开首次导入 scipy 的开销
    t0 = time.perf_counter()
    apply_rir(x, rir)
    dt_ms = (time.perf_counter() - t0) * 1000
    assert dt_ms < 50, f"卷混响耗时 {dt_ms:.0f} ms，说明退回了直接卷积（应 < 50 ms）"


def test_apply_rir_matches_direct_convolution():
    """FFT 卷积与直接卷积必须数值等价 —— 提速不能改变结果。"""
    x = RNG.standard_normal(SR)
    rir = make_rir(0.3, rng=RNG)
    fast = apply_rir(x, rir, align_direct=False)
    ref = np.convolve(x, rir)[: x.size]
    assert np.max(np.abs(fast - ref)) < 1e-9 * max(1.0, np.max(np.abs(ref)))


def test_apply_rir_aligns_direct_path():
    """对齐直达声后，混响信号与干信号的互相关峰应在 0 延迟处。

    不对齐会让 SI-SDR 白白损失好几 dB（纯时延被算成失真）。

    **测试信号刻意用白噪声而不是语音**：语音是准周期的（基频约 120 Hz，
    周期约 133 样本），它的自相关本身就有一串等高的周期性峰。
    再叠上混响尾巴在正延迟侧补充的能量，互相关的全局最大值很容易落到
    某个基频周期的整数倍上 —— 测出来的是**基频周期**，不是对齐误差。
    白噪声自相关是单峰的，才能真正测出延迟。
    """
    rir = make_rir(0.4, rng=RNG)
    x = RNG.standard_normal(SR)
    wet = apply_rir(x, rir, align_direct=True)
    xc = np.correlate(wet, x, mode="same")
    lag = int(np.argmax(np.abs(xc))) - SR // 2
    assert abs(lag) <= 2, f"直达声未对齐，残留延迟 {lag} 样本"

    # 同时直接验证裁剪量确实等于 RIR 主峰位置（直达声）
    unaligned = apply_rir(x, rir, align_direct=False)
    shift = int(np.argmax(np.abs(rir)))
    assert np.allclose(wet[: SR - shift], unaligned[shift:SR], atol=1e-12)


# ----------------------------------------------------------------- 噪声估计


def test_mcra_converges_to_true_noise_power():
    """纯噪声输入下，MCRA 的估计应收敛到真实噪声功率（误差 < 2 dB）。"""
    noise = make_noise("white", SR * 4, RNG) * 0.1
    est = MCRANoiseEstimator()
    sa = StreamingSTFT()
    cfg = DEFAULT_CONFIG
    n_fr = noise.size // cfg.hop
    for i in range(n_fr):
        est.update(np.abs(sa.push(noise[i * cfg.hop : (i + 1) * cfg.hop])) ** 2)
    # 与最后 1 秒的实际平均功率谱比较
    sa2 = StreamingSTFT()
    powers = [
        np.abs(sa2.push(noise[i * cfg.hop : (i + 1) * cfg.hop])) ** 2 for i in range(n_fr)
    ]
    true_power = np.mean(powers[-60:], axis=0)
    err_db = 10 * np.log10(np.mean(est.noise_power) / np.mean(true_power))
    assert abs(err_db) < 2.0, f"MCRA 噪声估计偏差 {err_db:.2f} dB"


def test_mcra_does_not_need_clean_leading_noise(speech):
    """信号一上来就是语音时，MCRA 仍应在 ~2 秒内收敛，而不是永久高估。

    这是它相对"前 N 帧当噪声"的核心优势，必须验证。
    """
    noise = make_noise("white", speech.size, RNG)
    noisy, scaled = mix_at_snr(speech, noise, 5.0, rng=RNG)
    est = MCRANoiseEstimator()
    sa = StreamingSTFT()
    cfg = DEFAULT_CONFIG
    n_fr = noisy.size // cfg.hop
    for i in range(n_fr):
        est.update(np.abs(sa.push(noisy[i * cfg.hop : (i + 1) * cfg.hop])) ** 2)
    true_noise_power = np.mean(scaled**2) * cfg.n_fft  # 粗略量级
    ratio_db = 10 * np.log10(np.mean(est.noise_power) / true_noise_power)
    assert abs(ratio_db) < 6.0, f"噪声估计偏离真值 {ratio_db:.1f} dB"


# ----------------------------------------------------------------- 增强器


def test_passthrough_is_exactly_transparent(speech):
    """增强器外壳本身不得引入任何误差，否则所有 Δ 指标都带系统偏差。"""
    out = PassThrough().process(speech)
    assert np.max(np.abs(out - speech)) < 1e-10


@pytest.mark.parametrize("method", ["specsub", "wiener", "mmse-lsa"])
def test_enhancers_improve_si_sdr_on_stationary_noise(speech, method):
    """平稳噪声下三种方法都必须有正收益。这是它们的主场，做不到就是实现有错。"""
    noisy, _ = mix_at_snr(speech, make_noise("white", speech.size, RNG), 0.0, rng=RNG)
    enh = build_dsp(method).process(noisy)
    gain = si_sdr(speech, enh) - si_sdr(speech, noisy)
    assert gain > 1.0, f"{method} 在白噪声 0 dB 下仅提升 {gain:.2f} dB"


@pytest.mark.parametrize("method", ["specsub", "wiener", "mmse-lsa"])
def test_enhancer_offline_equals_streaming(speech, method):
    """离线结果必须与逐帧流式结果逐样本相同。

    这条由 StreamingEnhancer.process() 的设计保证（离线就是用流式跑的），
    但仍然显式测一遍 —— 将来若有人为了提速加一条向量化的整段路径，
    这条测试会立刻抓住不一致。
    """
    from rtse.audio.stft import StreamingISTFT, num_frames

    noisy, _ = mix_at_snr(speech, make_noise("pink", speech.size, RNG), 5.0, rng=RNG)
    offline = build_dsp(method).process(noisy)

    cfg = DEFAULT_CONFIG
    n_fr = num_frames(noisy.size, cfg)
    feed = np.zeros(n_fr * cfg.hop)
    feed[: noisy.size] = noisy
    enh, sa, sy = build_dsp(method), StreamingSTFT(cfg), StreamingISTFT(cfg)
    enh.reset()
    streamed = np.concatenate(
        [sy.push(enh.process_frame(sa.push(feed[i * cfg.hop : (i + 1) * cfg.hop]))) for i in range(n_fr)]
    )[cfg.pad : cfg.pad + noisy.size]

    assert np.max(np.abs(offline - streamed)) < 1e-12


@pytest.mark.parametrize("method", ["specsub", "wiener", "mmse-lsa"])
def test_enhancer_reset_makes_runs_reproducible(speech, method):
    """reset() 后重跑必须得到完全相同的结果 —— Web 端热切换方法依赖这一点。"""
    noisy, _ = mix_at_snr(speech, make_noise("car", speech.size, RNG), 5.0, rng=RNG)
    enh = build_dsp(method)
    a = enh.process(noisy)
    b = enh.process(noisy)
    assert np.max(np.abs(a - b)) < 1e-15


@pytest.mark.parametrize("method", ["specsub", "wiener", "mmse-lsa"])
def test_enhancer_output_is_finite(speech, method):
    """静音、纯噪声、削波这些边界输入都不得产生 NaN/Inf。"""
    for sig in [np.zeros(SR), make_noise("white", SR, RNG), np.ones(SR) * 5.0]:
        out = build_dsp(method).process(sig)
        assert np.all(np.isfinite(out)), f"{method} 在边界输入下产生了非有限值"


# ----------------------------------------------------------------- VAD


@pytest.mark.parametrize("name", ["energy", "webrtc"])
def test_vad_detects_speech_and_silence(speech, name):
    """干净信号下，语音段应大部分判为语音，明确的静音段应大部分判为静音。"""
    vad = build_vad(name)
    flags = vad.process_signal(speech)
    cfg = DEFAULT_CONFIG

    def frames_in(t0, t1):
        return slice(int(t0 * SR / cfg.hop), int(t1 * SR / cfg.hop))

    # 静音段 1.0~1.6 s，留出挂起时间，只看 1.3~1.6 s
    silence_rate = flags[frames_in(1.3, 1.6)].mean()
    speech_rate = flags[frames_in(0.2, 0.9)].mean()
    assert speech_rate > 0.7, f"{name} 语音段检出率仅 {speech_rate:.2f}"
    assert silence_rate < 0.35, f"{name} 静音段误触发率达 {silence_rate:.2f}"


@pytest.mark.parametrize("name", ["energy", "webrtc"])
def test_vad_all_silence_gives_no_speech(name):
    assert not build_vad(name).process_signal(np.zeros(SR * 2)).any()


def test_webrtc_vad_handles_frame_size_mismatch():
    """hop=256 不是 WebRTC 允许的帧长，适配层必须把它处理掉而不是抛异常。"""
    vad = build_vad("webrtc")
    assert DEFAULT_CONFIG.hop not in (160, 320, 480)  # 前提成立才有测试价值
    for _ in range(50):
        vad.process(RNG.standard_normal(DEFAULT_CONFIG.hop) * 0.1)


def test_webrtc_vad_does_not_overflow_on_loud_input():
    """|x| > 1 的输入必须限幅而不是整数溢出绕回（会产生随机错误判决）。"""
    vad = build_vad("webrtc")
    for _ in range(20):
        vad.process(np.ones(DEFAULT_CONFIG.hop) * 3.0)  # 不抛异常即通过


# ----------------------------------------------------------------- 管线


def test_pipeline_passthrough_is_transparent(speech):
    out, frames = Pipeline().process_signal(speech)
    assert np.max(np.abs(out - speech)) < 1e-10
    assert len(frames) > 0


def test_pipeline_matches_bare_enhancer(speech):
    """管线加了 VAD 和计量，但不开门控时，输出必须与裸增强器完全相同。"""
    noisy, _ = mix_at_snr(speech, make_noise("white", speech.size, RNG), 5.0, rng=RNG)
    bare = build_dsp("wiener").process(noisy)
    piped, _ = Pipeline(enhancer=build_dsp("wiener"), vad=build_vad("energy")).process_signal(noisy)
    assert np.max(np.abs(bare - piped)) < 1e-12


def test_pipeline_latency_stats_are_sane(speech):
    pipe = Pipeline(enhancer=build_dsp("wiener"), vad=build_vad("energy"))
    pipe.process_signal(speech)
    st = pipe.latency_stats()
    assert st.n_frames > 0
    assert st.frame_budget_ms == pytest.approx(16.0, abs=0.01)
    assert st.algorithmic_latency_ms == pytest.approx(32.0, abs=0.01)
    assert 0 < st.p50_ms <= st.p95_ms <= st.p99_ms <= st.max_ms
    assert st.rtf > 0
    d = st.as_dict()
    assert set(d) >= {"rtf", "proc_p99_ms", "realtime_ok", "algorithmic_latency_ms"}


def test_pipeline_vad_gate_attenuates_silence(speech):
    """开启门控后，静音段能量应被明显压低，语音段基本不受影响。"""
    noisy, _ = mix_at_snr(speech, make_noise("white", speech.size, RNG), 10.0, rng=RNG)
    ungated, _ = Pipeline(enhancer=build_dsp("wiener"), vad=build_vad("energy")).process_signal(noisy)
    gated, _ = Pipeline(
        enhancer=build_dsp("wiener"), vad=build_vad("energy"), vad_gate=True
    ).process_signal(noisy)

    sil = slice(int(1.3 * SR), int(1.6 * SR))
    spk = slice(int(0.3 * SR), int(0.9 * SR))
    sil_ratio = np.sqrt(np.mean(gated[sil] ** 2)) / (np.sqrt(np.mean(ungated[sil] ** 2)) + 1e-12)
    spk_ratio = np.sqrt(np.mean(gated[spk] ** 2)) / (np.sqrt(np.mean(ungated[spk] ** 2)) + 1e-12)
    assert sil_ratio < 0.5, f"门控未压低静音段（比值 {sil_ratio:.2f}）"
    assert spk_ratio > 0.9, f"门控误伤了语音段（比值 {spk_ratio:.2f}）"
