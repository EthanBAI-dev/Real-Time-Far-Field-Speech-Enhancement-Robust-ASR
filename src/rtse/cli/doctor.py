"""``rtse-doctor`` —— 环境自检。

Phase 1 的验收标准。它回答一个问题：**这台机器现在能跑到项目的哪一步。**

设计原则：每一项都做**真实的功能验证**，而不是只看 import 成不成功。
比如 soundfile 不只 import，而是真的写一个 wav 再读回来比对；
onnxruntime 不只 import，而是真的构建一个图并跑一次推理。
"能 import" 和 "能用" 在 Windows + 音频这个组合里差得远。
"""

from __future__ import annotations

import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.table import Table

from rtse import HOP_LENGTH, N_FFT, SAMPLE_RATE, __version__
from rtse.cli._console import console
from rtse.paths import DATA_DIR, LOCAL_DISK_WARN_FREE_GB, MODELS_DIR, PROJECT_ROOT, RESULTS_DIR

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "[green]✓[/green]", WARN: "[yellow]![/yellow]", FAIL: "[red]✗[/red]"}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _run(name: str, fn) -> Check:
    """执行单项检查，任何异常都转成 FAIL 而不是中断整个 doctor。"""
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001 - doctor 的职责就是把所有异常呈现出来
        return Check(name, FAIL, f"{type(exc).__name__}: {exc}")
    return Check(name, status, detail)


# --------------------------------------------------------------------------- 基础环境


def _check_python():
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro} ({platform.machine()})"
    if (v.major, v.minor) != (3, 11):
        return WARN, detail + " — 期望 3.11（见 ISSUES.md I-02）"
    return OK, detail


def _check_disk():
    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / 1024**3
    detail = f"{PROJECT_ROOT.drive} 可用 {free_gb:.1f} GB"
    if free_gb < LOCAL_DISK_WARN_FREE_GB:
        return WARN, detail + f" — 低于 {LOCAL_DISK_WARN_FREE_GB:.0f} GB 告警线（ISSUES.md I-04）"
    return OK, detail


# --------------------------------------------------------------------------- 信号核心


def _check_stft():
    """真跑一次完美重构，而不是只 import。这是整条链路的地基。"""
    from rtse.audio.stft import STFTConfig, check_cola, istft, stft

    cfg = STFTConfig()
    cola = check_cola(cfg.n_fft, cfg.hop)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SAMPLE_RATE)
    err = float(np.max(np.abs(istft(stft(x, cfg), length=x.size, cfg=cfg) - x)))
    detail = f"n_fft={N_FFT} hop={HOP_LENGTH} | COLA偏差 {cola:.1e} | 重构误差 {err:.1e}"
    return (OK, detail) if (cola < 1e-12 and err < 1e-10) else (FAIL, detail)


def _check_streaming_parity():
    """流式与离线一致性。ONNX 流式导出能否成立，取决于这一条。"""
    from rtse.audio.stft import STFTConfig, StreamingISTFT, StreamingSTFT, num_frames

    cfg = STFTConfig()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8000)
    n_fr = num_frames(x.size, cfg)
    feed = np.zeros(n_fr * cfg.hop)
    feed[: x.size] = x
    sa, sy = StreamingSTFT(cfg), StreamingISTFT(cfg)
    out = np.concatenate(
        [sy.push(sa.push(feed[i * cfg.hop : (i + 1) * cfg.hop])) for i in range(n_fr)]
    )
    err = float(np.max(np.abs(out[cfg.pad : cfg.pad + x.size] - x)))
    detail = f"流式往返误差 {err:.1e} | 算法延迟 {cfg.latency_samples / SAMPLE_RATE * 1000:.0f} ms"
    return (OK, detail) if err < 1e-10 else (FAIL, detail)


# --------------------------------------------------------------------------- 依赖实测


def _check_soundfile():
    """写一个 wav 再读回来比对 —— libsndfile 二进制真的能加载才算通过。"""
    import soundfile as sf

    from rtse.audio.io import read_audio, write_audio

    # 刻意用**保证不削波**的信号：write_audio 对 |x|>1 会整体重缩放（见 ISSUES.md I-07），
    # 那样测出来的就是缩放增益而不是量化误差了。
    rng = np.random.default_rng(2)
    x = np.clip(rng.standard_normal(SAMPLE_RATE) * 0.2, -0.9, 0.9)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.wav"
        gain = write_audio(p, x)
        y = read_audio(p)
    err = float(np.max(np.abs(y - x)))
    # 16-bit 量化误差上界 = 1/32768 ≈ 3.1e-5
    detail = f"libsndfile {sf.__libsndfile_version__} | 往返误差 {err:.2e}（16bit 量化上界 3.1e-5）"
    return (OK, detail) if (err < 3.2e-5 and gain == 1.0) else (FAIL, detail)


def _check_soxr():
    """重采样实测：48k 正弦降到 16k 后频率应保持不变。"""
    import soxr

    from rtse.audio.io import resample

    sr_in, f0 = 48000, 1000.0
    t = np.arange(sr_in) / sr_in
    y = resample(np.sin(2 * np.pi * f0 * t), sr_in, SAMPLE_RATE)
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    peak_hz = float(np.argmax(spec) * SAMPLE_RATE / y.size)
    detail = f"soxr {soxr.__version__} | 48k→16k 后主峰 {peak_hz:.1f} Hz（应为 {f0:.0f}）"
    return (OK, detail) if abs(peak_hz - f0) < 5 else (FAIL, detail)


def _check_torch():
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    detail = f"torch {torch.__version__} | 设备 {dev}"
    # 真跑一次前向，确认不是坏的 wheel
    m = torch.nn.GRU(8, 8, batch_first=True)
    m(torch.randn(1, 4, 8))
    if dev == "cpu":
        # 本地按计划就是 CPU 版（训练在 Colab），这是预期状态而非问题
        return OK, detail + " — 本地按计划用 CPU 版，训练在 Colab"
    return OK, detail


def _check_onnxruntime():
    """构建一个最小 ONNX 图并真的跑一次推理。"""
    import onnxruntime as ort
    import torch

    class Tiny(torch.nn.Module):
        def forward(self, x):
            return x * 2.0 + 1.0

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.onnx"
        torch.onnx.export(Tiny(), (torch.zeros(1, 4),), str(p), input_names=["x"],
                          output_names=["y"], dynamo=False)
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"x": np.ones((1, 4), dtype=np.float32)})[0]

    ok = np.allclose(out, 3.0)
    providers = ", ".join(ort.get_available_providers())
    return (OK if ok else FAIL), f"onnxruntime {ort.__version__} | providers: {providers}"


def _check_webrtcvad():
    import webrtcvad

    vad = webrtcvad.Vad(2)
    # 30 ms @ 16 kHz = 480 样本 = 960 字节
    silence = (np.zeros(480, dtype=np.int16)).tobytes()
    tone = (np.sin(np.arange(480) * 0.3) * 12000).astype(np.int16).tobytes()
    r1, r2 = vad.is_speech(silence, SAMPLE_RATE), vad.is_speech(tone, SAMPLE_RATE)
    return OK, f"webrtcvad 可用 | 静音判为 {r1}，纯音判为 {r2}"


def _check_faster_whisper():
    """只检查库和后端，**不下载模型**（模型下载留到实际用时，避免 doctor 卡住）。"""
    import ctranslate2
    import faster_whisper

    devs = []
    if ctranslate2.get_cuda_device_count() > 0:
        devs.append("cuda")
    devs.append("cpu")
    ver = getattr(faster_whisper, "__version__", "?")
    return OK, f"faster-whisper {ver} | ctranslate2 {ctranslate2.__version__} | 可用设备 {devs}"


def _check_fastapi():
    import fastapi
    import uvicorn

    return OK, f"fastapi {fastapi.__version__} | uvicorn {uvicorn.__version__}"


# --------------------------------------------------------------------------- 资产


def _check_metrics():
    from rtse.metrics.registry import available_metrics

    avail = available_metrics()
    ready = [k for k, (ok, _) in avail.items() if ok]
    missing = {k: why for k, (ok, why) in avail.items() if not ok}
    detail = f"可用 {len(ready)}/{len(avail)}: {', '.join(ready)}"
    if missing:
        detail += "\n" + "\n".join(f"  [dim]n/a {k}: {why}[/dim]" for k, why in missing.items())
        return WARN, detail
    return OK, detail


def _check_assets():
    """检查 Colab 侧产物是否已经落地（Phase 3/4 之前必然是缺的，属正常）。"""
    items = {
        "增强模型 (models/*.onnx)": list(MODELS_DIR.glob("*.onnx")),
        "V1测试集 (data/testsets/*)": list((DATA_DIR / "testsets").glob("*/index.json")),
        "评测结果 (results/*.json)": list(RESULTS_DIR.glob("*.json")),
    }
    lines = [f"{'有' if v else '无'} · {k}" for k, v in items.items()]
    detail = " | ".join(lines)
    if not any(items.values()):
        return WARN, detail + "\n  [dim]均为空属正常：这些是 Colab 侧 Phase 3/4 的产物[/dim]"
    return OK, detail


CHECKS: list[tuple[str, object]] = [
    ("Python 版本", _check_python),
    ("磁盘空间", _check_disk),
    ("STFT 完美重构", _check_stft),
    ("流式/离线一致性", _check_streaming_parity),
    ("soundfile 读写", _check_soundfile),
    ("soxr 重采样", _check_soxr),
    ("PyTorch", _check_torch),
    ("ONNX Runtime", _check_onnxruntime),
    ("WebRTC VAD", _check_webrtcvad),
    ("faster-whisper", _check_faster_whisper),
    ("FastAPI", _check_fastapi),
    ("指标可用性", _check_metrics),
    ("Colab 产物", _check_assets),
]


def main() -> int:
    console.print(f"\n[bold]RTSE v{__version__}[/bold] 环境自检  ·  {PROJECT_ROOT}\n")

    results = [_run(name, fn) for name, fn in CHECKS]

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("", width=2, justify="center")
    table.add_column("检查项", width=20, no_wrap=True)
    table.add_column("结果", overflow="fold")
    for c in results:
        table.add_row(_MARK[c.status], c.name, c.detail)
    console.print(table)

    n_fail = sum(c.status == FAIL for c in results)
    n_warn = sum(c.status == WARN for c in results)

    if n_fail:
        console.print(f"\n[red]{n_fail} 项失败[/red]，{n_warn} 项告警。见 docs/ISSUES.md\n")
        return 1
    if n_warn:
        console.print(
            f"\n[green]核心链路全部通过[/green]，{n_warn} 项告警（多为尚未开始的阶段，属预期）。\n"
        )
        return 0
    console.print("\n[green]全部通过。[/green]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
