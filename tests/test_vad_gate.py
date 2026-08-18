"""`rtse-eval` 的 VAD 门控开关（docs/ISSUES.md I-34）。

**由来**：DSP/神经网络增强器的 `process_frame(spec)` 接口本身不接受 VAD 输入——
`rtse-eval` 此前虽然构造了 VAD 并跑了一遍，但从未把 `vad_gate=True` 传给
`Pipeline`，VAD 判决只是被塞进 `FrameResult` 供诊断，从没接触过增强后的频谱。
历史上所有 DSP 方法的 CER 数字，都是在 VAD 完全不参与增强的情况下跑出来的。

这里测两件事：`_make_pipeline` 确实把 `--vad`/`--vad-gate` 传到了 Pipeline 上
（防止再出现"参数加了但没接上"），以及打开门控之后静音段真的被压低了
（防止 Pipeline 内部的门控逻辑本身失效导致测试对了但行为没变）。
"""

import numpy as np
import pytest

from rtse.cli.evaluate import _make_pipeline
from rtse.dsp import build_dsp
from rtse.vad.energy import EnergyVAD
from rtse.vad.webrtc import WebRTCVAD

SR = 16000


def _speech_then_silence(n=SR * 2):
    """前半段类语音，后半段纯静音——用来看门控有没有把静音段压下去。"""
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 200 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    x[n // 2 :] = 0.0
    return x.astype(np.float64)


def test_make_pipeline_wires_vad_choice_and_gate_flag():
    p_energy = _make_pipeline(None, "energy", False, -20.0)
    assert isinstance(p_energy.vad, EnergyVAD)
    assert p_energy.vad_gate is False

    p_webrtc_gated = _make_pipeline(None, "webrtc", True, -18.0)
    assert isinstance(p_webrtc_gated.vad, WebRTCVAD)
    assert p_webrtc_gated.vad_gate is True
    assert np.isclose(p_webrtc_gated.gate_floor, 10.0 ** (-18.0 / 20.0))


def test_gate_off_leaves_silence_at_dsp_output_level():
    """门控关闭时（历史行为），静音段的输出电平只由 DSP 本身决定，
    不会被额外压低——这是回归防线：不能让默认行为悄悄变了。
    """
    x = _speech_then_silence()
    noise = 0.05 * np.random.default_rng(0).standard_normal(x.size)
    noisy = x + noise

    off = _make_pipeline(build_dsp("wiener"), "energy", False, -20.0)
    out_off, _ = off.process_signal(noisy)

    # 静音段输出应约等于噪声本身被 DSP 处理后的残留，而不是被压到底噪之下
    tail = out_off[-SR // 4 :]
    assert np.sqrt(np.mean(tail**2)) > 1e-4


def test_gate_on_suppresses_silence_more_than_gate_off():
    """打开门控后，静音段（已知没有语音的后半段）的残留能量必须
    明显低于不开门控的情况——否则门控参数虽然传进去了，行为却没生效。
    """
    x = _speech_then_silence()
    noise = 0.05 * np.random.default_rng(0).standard_normal(x.size)
    noisy = x + noise

    off = _make_pipeline(build_dsp("wiener"), "energy", False, -20.0)
    on = _make_pipeline(build_dsp("wiener"), "energy", True, -20.0)
    out_off, _ = off.process_signal(noisy)
    out_on, _ = on.process_signal(noisy)

    tail = slice(-SR // 4, None)
    rms_off = np.sqrt(np.mean(out_off[tail] ** 2))
    rms_on = np.sqrt(np.mean(out_on[tail] ** 2))
    assert rms_on < rms_off * 0.5, (
        f"门控应显著压低静音段残留：off={rms_off:.4f} on={rms_on:.4f}"
    )


def test_eval_cli_vad_choices_cover_every_registered_vad():
    """`--vad` 的可选值必须来自 VAD 注册表，不能是手写清单。

    第一版写死了 ('energy','webrtc')，后来新增 SileroVAD 时没同步，
    `--vad silero` 被 argparse 直接拒掉——而报错长得像"这个 VAD 不存在"，
    实际上它早就注册好了。跟 I-29 是同一类：两处各自演进、中间靠手写清单耦合。
    """
    import contextlib
    import io

    from rtse.cli.evaluate import main
    from rtse.vad import VAD_METHODS

    # 用 --help 的实际输出来断言，比反射 parser 更贴近用户看到的东西
    buf = io.StringIO()
    with (
        contextlib.redirect_stdout(buf),
        contextlib.suppress(SystemExit),
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setattr("sys.argv", ["rtse-eval", "--help"])
        main()
    help_text = buf.getvalue()
    assert "--vad" in help_text, "帮助里没有 --vad，参数可能被删了"
    for name in VAD_METHODS:
        assert name in help_text, (
            f"VAD {name!r} 已注册但 --vad 的 choices 里没有——"
            "choices 应当从 VAD_METHODS 动态生成"
        )
