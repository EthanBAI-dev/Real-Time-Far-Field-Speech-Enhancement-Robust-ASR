"""DSP 增强器、VAD、管线的测试。"""

from pathlib import Path

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


def test_mix_at_snr_removes_noise_dc_before_scaling():
    """真实噪声带 DC 偏置时，目标和实测 SNR 仍必须一致（I-32）。"""
    rng = np.random.default_rng(32)
    speech_sig = rng.standard_normal(SR * 2) * 0.2
    noise = rng.standard_normal(SR * 2) * 0.05 + 0.4
    mixed, scaled_noise = mix_at_snr(speech_sig, noise, 5.0, rng=rng)

    signal = speech_sig - speech_sig.mean()
    mask = speech_active_mask(signal)
    actual = 10 * np.log10(
        np.mean(signal[mask] ** 2) / np.mean((scaled_noise - scaled_noise.mean()) ** 2)
    )
    assert abs(actual - 5.0) < 1e-6
    assert abs(float(scaled_noise.mean())) < 1e-12
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


def test_mcra_dual_timescale_never_increases_estimate():
    """I-20 修复的核心保证：双时间尺度地板 = min(短窗口, 长窗口)，
    数学上**只可能更低或相等，不可能比单窗口版本更高**。

    这条测试不依赖任何具体音频场景（构造反例极其困难，见 I-20 的详细记录——
    真实 bug 需要"连续真实语音 + 混响 + 数秒时长"才会显现，玩具信号里
    长短窗口经常给出完全相同的结果，测不出差异）。改为直接验证这个不变量：
    换噪声估计器不会让原本正常的场景变差，这是这次修复"安全"的真正原因，
    比端到端 SI-SDR 更可靠、更不依赖具体音频素材。
    """
    rng = np.random.default_rng(55)
    x = rng.standard_normal(SR * 3) * 0.3
    cfg = DEFAULT_CONFIG
    sa = StreamingSTFT(cfg)
    single = MCRANoiseEstimator(cfg, L2_mult=1)  # 等价于修复前（长窗口=短窗口）
    dual = MCRANoiseEstimator(cfg, L2_mult=8)
    n_fr = x.size // cfg.hop
    for i in range(n_fr):
        power = np.abs(sa.push(x[i * cfg.hop : (i + 1) * cfg.hop])) ** 2
        single.update(power)
        dual.update(power)
        assert np.all(dual.noise_power <= single.noise_power + 1e-9), (
            f"第 {i} 帧：双时间尺度估计值超过了单窗口版本，违反了 min() 的数学保证"
        )


@pytest.mark.xfail(
    reason="I-20：混响+高SNR的主要失效模式已修复（双时间尺度地板），"
    "但短促静音间隙场景仍未解决——已确认根因不是 alpha_d 收敛速度"
    "（调低 alpha_d 反而让这条测试更差，见 ISSUES.md I-20 的参数网格搜索记录），"
    "需要比调参更深层的改动才能解决",
    strict=True,
)
def test_mcra_high_snr_known_bug(speech):
    """已知未解决的 bug（docs/ISSUES.md I-20）。

    用真实 Colab 评测数据发现：SNR>=15 dB 时 wiener 处理后的 SI-SDR
    **集体劣于不处理**，且 20/20 样本一致、跨噪声类型一致，不是离群点。
    追踪到 MCRA 内部：SNR=20 时真实噪声功率 5.8e-4，估计出来是 24.97，
    差了约 4.3 万倍——语音帧的 ratio 长时间够不到 delta=5.0 的判决门槛，
    被持续误判成噪声、缓慢吸收进噪声估计。

    这条测试**故意**标记 `xfail(strict=True)`：现在必须是红的，
    这样它才能诚实地反映"这个问题还没修"。修好 MCRA 后这条会意外变绿，
    `strict=True` 下 xfail 却 pass 本身会被判为失败，逼着去把标记摘掉——
    比"改完感觉应该好了"更可靠的确认方式。

    不要因为这条测试失败就去调整 delta/alpha_s/L 草草糊过去：
    这几个参数相互耦合，随手改一个大概率只是把问题挪到别的 SNR/噪声条件下，
    需要系统性重新验证才能真正解决（原因见 ISSUES.md I-20）。

    用**独立的本地 RNG**，不用模块级共享的 `RNG`——本文件里所有测试共用
    同一个 `RNG` 对象，谁多消耗几次抽取，后面的测试拿到的随机序列就会跟着变，
    新增测试插在中间会让排在它后面的测试意外失败。这条踩过一次了。
    """
    local_rng = np.random.default_rng(20260820)
    noise = make_noise("white", speech.size, local_rng)
    noisy, _ = mix_at_snr(speech, noise, 20.0, rng=local_rng)
    enh = build_dsp("wiener").process(noisy)
    gain = si_sdr(speech, enh) - si_sdr(speech, noisy)
    assert gain > 0, f"SNR=20 dB 下 wiener 处理后 SI-SDR 反而下降了 {-gain:.2f} dB"


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


@pytest.mark.parametrize("name", ["energy", "webrtc", "silero"])
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


def test_silero_vad_requires_hop_to_divide_its_fixed_chunk():
    """Silero 只接受固定 512 样本的块；hop 必须整除它，缓冲逻辑才成立。

    hop=192（配 n_fft=384 才能通过 STFTConfig 自己的 COLA 校验）不整除 512，
    专门用来触发这条防线，跟项目实际使用的 hop=256 默认配置无关。
    """
    from rtse.audio.stft import STFTConfig
    from rtse.vad.silero import SileroVAD

    bad_cfg = STFTConfig(n_fft=384, hop=192)
    assert 512 % bad_cfg.hop != 0  # 前提成立才有测试价值
    with pytest.raises(ValueError, match="整除"):
        SileroVAD(bad_cfg)


def test_silero_vad_survives_many_hops_without_crashing():
    """hop=256 是 512 的整数倍，缓冲每两次 process() 触发一次真实推理——
    多跑几百帧，确认缓冲边界不会累积出错或抛异常。"""
    vad = build_vad("silero")
    for _ in range(500):
        vad.process(RNG.standard_normal(DEFAULT_CONFIG.hop) * 0.1)


def test_silero_vad_prob_in_unit_interval():
    vad = build_vad("silero")
    for _ in range(30):
        vf = vad.process(RNG.standard_normal(DEFAULT_CONFIG.hop) * 0.2)
        assert 0.0 <= vf.prob <= 1.0


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "ref").exists(),
    reason="需要本地的 ref/speech-processing-master 参考语料，不在仓库里",
)
def test_silero_vad_detects_real_speech_recording():
    """合成的正弦谐波信号骗不过神经网络 VAD——手动验证过 Silero 对着这类
    "假语音"给出的概率反而低于纯噪声（真实、符合预期：它是在真实录音上训练的，
    没见过这种信号）。所以这条必须用真实录音测，而不是像 EnergyVAD/WebRTCVAD
    那样能用合成信号糊弄过去；真实语料不在仓库里，本地没有就跳过。
    """
    import soundfile as sf

    ref_dir = Path(__file__).resolve().parents[1] / "ref" / "speech-processing-master"
    wav = next(ref_dir.rglob("sf1_cln.wav"), None) or next(ref_dir.rglob("*cln*.wav"), None)
    if wav is None:
        pytest.skip("ref/ 下找不到示例语音")
    y, sr = sf.read(wav)
    assert sr == SR

    decisions = build_vad("silero").process_signal(y)
    assert decisions.mean() > 0.7, f"真实语音判为语音的比例只有 {decisions.mean():.2f}"


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


def test_mmse_lsa_defaults_to_fast_poly_approx():
    """默认值是 False（走多项式近似）——见 F-04，已验证不影响降噪质量。"""
    from rtse.dsp.enhancers import MMSELogSTSA

    assert MMSELogSTSA().exact is False


def test_exp1_half_poly_matches_exact_within_bound():
    """多项式近似的单点误差要在已知界内。

    这条测过一次假阳性：第一版实现漏了一个 0.5 因子，端到端 SI-SDR 测试量出
    2.5dB 的"回退"，一度以为是 decision-directed 反馈环把误差放大了。
    补上那个因子后，单点误差和端到端结果才对上号——两者本该一致，
    只是被抄写 bug 短暂地弄得像是两回事。
    """
    from scipy.special import exp1

    from rtse.dsp.enhancers import _exp1_half_poly

    nu = np.geomspace(1e-6, 500, 200)
    exact = 0.5 * exp1(nu)
    approx = _exp1_half_poly(nu)
    err_db = 20 * np.log10(np.exp(exact)) - 20 * np.log10(np.exp(approx))
    assert np.max(np.abs(err_db)) < 0.6, f"最大增益误差 {np.max(np.abs(err_db)):.2f} dB 超出已知界"


@pytest.mark.parametrize(
    "kind,snr",
    [(k, s) for k in ("white", "babble", "keyboard") for s in (-5, 0, 5, 10, 15)],
)
def test_mmse_lsa_poly_approx_matches_exact_si_sdr(kind, snr):
    """近似解不能只在单一条件下测——15 组噪声类型 × SNR 组合都要接近精确解，
    否则"整体安全"这个结论只是挑对了一个巧合的测试点。
    """
    from rtse.dsp.enhancers import MMSELogSTSA

    n = SR * 4
    t = np.arange(n) / SR
    f0 = 110 + 30 * np.sin(2 * np.pi * RNG.uniform(0.8, 1.8) * t)
    sig = sum(np.sin(h * 2 * np.pi * np.cumsum(f0) / SR) / h for h in range(1, 12))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * RNG.uniform(3, 5) * t)
    env[int(0.8 * SR) : int(1.3 * SR)] = 0
    env[int(2.5 * SR) : int(3.0 * SR)] = 0
    speech_sig = (sig * env) / np.max(np.abs(sig * env)) * 0.6

    noisy, _ = mix_at_snr(speech_sig, make_noise(kind, n, RNG), snr, rng=RNG)
    gain_exact = si_sdr(speech_sig, MMSELogSTSA(exact=True).process(noisy)) - si_sdr(speech_sig, noisy)
    gain_approx = si_sdr(speech_sig, MMSELogSTSA(exact=False).process(noisy)) - si_sdr(speech_sig, noisy)
    assert abs(gain_exact - gain_approx) < 0.5, (
        f"{kind}@{snr}dB：精确解收益 {gain_exact:.3f}dB，近似解 {gain_approx:.3f}dB，"
        f"差距 {gain_exact - gain_approx:+.3f}dB 超出预期范围"
    )


def test_backoff_is_off_for_specsub_on_for_the_other_two():
    """谱减法不该开收手混合——它的过减因子本来就随帧 SNR 自适应，
    高 SNR 段实测没有负收益（+0.12 dB）。重复施加等于处理两次。"""
    assert build_dsp("specsub").backoff_snr_db is None
    assert build_dsp("wiener").backoff_snr_db is not None
    assert build_dsp("mmse-lsa").backoff_snr_db is not None


def test_backoff_leaves_clean_input_nearly_untouched(speech):
    """干净输入进来，收手机制要让维纳/MMSE-LSA 基本透传。

    修复前实测 clean 透传 32.2 dB（维纳），修复后 47.5 dB。
    这条钉的是"收手确实生效"，不是具体数值——数值随语料变。
    """
    from rtse.dsp.enhancers import MMSELogSTSA, WienerFilter

    for cls in (WienerFilter, MMSELogSTSA):
        off = si_sdr(speech, cls(backoff_snr_db=None).process(speech))
        on = si_sdr(speech, cls().process(speech))
        assert on > off + 3.0, (
            f"{cls.__name__} 收手后干净输入应明显少受损：关={off:.1f}dB 开={on:.1f}dB"
        )


@pytest.mark.parametrize("cls_name", ["wiener", "mmse-lsa"])
def test_backoff_does_not_sacrifice_low_snr_gain(speech, cls_name):
    """收手只该在噪声已经很少时起作用；低 SNR 段的降噪收益不能被削掉。

    这是扫参时的核心约束：(0,12) 那档高 SNR 更好，但 mmse-lsa 的低 SNR 收益
    从 +0.52 掉到 +0.41，所以最终选了 (2,14)。
    """
    from rtse.dsp import build_dsp as _b
    from rtse.dsp.enhancers import MMSELogSTSA, WienerFilter

    cls = WienerFilter if cls_name == "wiener" else MMSELogSTSA
    noisy, _ = mix_at_snr(speech, make_noise("white", speech.size, RNG), 0.0, rng=RNG)
    base = si_sdr(speech, noisy)
    off = si_sdr(speech, cls(backoff_snr_db=None).process(noisy)) - base
    on = si_sdr(speech, _b(cls_name).process(noisy)) - base
    assert on > off - 0.3, f"0 dB 下收手削掉了太多收益：关={off:+.2f}dB 开={on:+.2f}dB"
