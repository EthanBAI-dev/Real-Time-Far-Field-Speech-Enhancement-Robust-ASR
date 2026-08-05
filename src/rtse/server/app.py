"""FastAPI 服务：实时演示 + 文件实验台 + 基准看板。

**为什么麦克风采集放在浏览器而不是服务端**：
服务端用 sounddevice/PortAudio 直接抓麦克风，在 Windows 上要面对
WASAPI/DirectSound/MME 三套后端、独占模式、采样率协商、设备热插拔一堆问题，
而且演示时只能在跑服务的那台机器上听。浏览器的 WebAudio 把这些全接管了，
还顺带获得跨机器演示的能力。代价是多一次 WebSocket 往返 —— 这部分延迟单独测量、单独呈现。

**WebSocket 协议**（每个音频块两条消息，WS 保序）：
- 上行：二进制 Float32Array（16 kHz 单声道 PCM）；文本 JSON 为控制指令
- 下行：先发文本 JSON（元数据/频谱/指标），再发二进制 Float32Array（增强后音频）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rtse import HOP_LENGTH, N_FFT, SAMPLE_RATE, __version__
from rtse.paths import PROJECT_ROOT, RESULTS_DIR
from rtse.server.session import SPEC_BINS, SessionConfig, StreamSession

WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(title="RTSE", version=__version__)


@app.middleware("http")
async def no_stale_assets(request, call_next):
    """给页面与静态资源加 ``Cache-Control: no-cache``。

    **为什么必须显式加**：Starlette 的 ``StaticFiles`` 只发 ``ETag`` 和
    ``Last-Modified``，**不发** ``Cache-Control``。缺了它，浏览器会启用
    *启发式缓存* —— 按 Last-Modified 的时间差估一个过期时间，
    在此期间**连条件请求都不发**，直接用本地副本。

    后果是改完前端刷新页面毫无变化，而且症状极具迷惑性：
    新的 HTML 配上缓存的旧 JS，页面看着变了、行为却没变
    （见 docs/ISSUES.md I-17，这个坑实际踩到了）。

    用 ``no-cache`` 而不是 ``no-store``：前者只是强制**每次都验证**，
    ETag 未变时服务端仍回 304，几乎没有额外开销；
    后者会彻底禁用缓存，连 304 的机会都没有。
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path in ("/", "/lab", "/bench"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


# --------------------------------------------------------------------- 元信息


@app.get("/api/info")
def api_info() -> JSONResponse:
    """前端启动时拉取的能力清单。缺什么、为什么缺，都如实告诉界面。"""
    from rtse.dsp import DSP_METHODS
    from rtse.metrics.registry import available_metrics
    from rtse.paths import MODELS_DIR
    from rtse.vad import VAD_METHODS

    nn_models = sorted(p.stem for p in MODELS_DIR.glob("*.onnx"))
    metrics = {k: {"ok": ok, "why": why} for k, (ok, why) in available_metrics().items()}
    return JSONResponse(
        {
            "version": __version__,
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop": HOP_LENGTH,
            "spec_bins": SPEC_BINS,
            "frame_ms": HOP_LENGTH / SAMPLE_RATE * 1000.0,
            "algo_latency_ms": N_FFT / SAMPLE_RATE * 1000.0,
            "dsp_methods": sorted(DSP_METHODS),
            "nn_models": nn_models,
            "vad_methods": sorted(VAD_METHODS),
            "metrics": metrics,
            # 神经模型尚未从 Colab 拿回来时，界面上明确显示"待 Colab 产出"而不是假装没这回事
            "nn_ready": bool(nn_models),
        }
    )


@app.get("/api/demo/clips")
def api_demo_clips() -> JSONResponse:
    """列出可用的演示音频，供「虚拟麦克风」使用。

    存在的理由：这台机器没有可用的麦克风端点（见 docs/ISSUES.md I-16），
    而且作品集演示时也未必总有麦克风。虚拟麦克风把音频文件按实时节奏
    喂进**完全相同**的 WebSocket 路径，所有实时可视化与延迟指标照常工作。
    """
    from rtse.paths import DATA_DIR

    demo = DATA_DIR / "demo"
    clips = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        for p in sorted(demo.glob(ext)):
            clips.append({"id": p.stem, "name": p.name})
    return JSONResponse({"clips": clips})


@app.get("/api/demo/mixed")
def api_demo_mixed(
    clip: str = "", noise: str = "babble", snr_db: float = 5.0, t60: float = 0.0,
    seconds: float = 20.0,
) -> Response:
    """按指定条件合成带噪音频，返回 16 kHz 单声道 WAV。

    合成放在**服务端**而不是浏览器，是为了让"0 dB"在虚拟麦克风、文件实验台、
    Colab 训练三处是**同一个定义** —— 都走 `mix_at_snr`（只在语音活跃段上定义 SNR）。
    在前端另写一套混音，两边的 SNR 迟早会对不上。
    """
    import io

    import soundfile as sf

    from rtse.audio.io import read_audio
    from rtse.data.synth import apply_rir, make_noise, make_rir, mix_at_snr
    from rtse.paths import DATA_DIR

    demo = DATA_DIR / "demo"
    candidates = [p for ext in ("*.wav", "*.flac", "*.mp3") for p in sorted(demo.glob(ext))]
    if clip:
        candidates = [p for p in candidates if p.stem == clip] or candidates
    if not candidates:
        return JSONResponse({"error": f"data/demo/ 下没有音频文件（找了 {demo}）"}, status_code=404)

    rng = np.random.default_rng(0)  # 固定种子 → 同样的参数每次得到同样的音频
    x = read_audio(candidates[0])[: int(seconds * SAMPLE_RATE)]
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak * 0.7

    wet = apply_rir(x, make_rir(t60, rng=rng)) if t60 > 0 else x
    if noise and noise != "none":
        y, _ = mix_at_snr(wet, make_noise(noise, wet.size, rng), snr_db, rng=rng)
    else:
        y = wet

    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/api/lab/run")
async def api_lab_run(
    file: UploadFile | None = File(None),
    noise_kind: str = Form("babble"),
    snr_db: float = Form(5.0),
    t60: float = Form(0.0),
    methods: str = Form("none,specsub,wiener,mmse-lsa"),
    vad: str = Form("energy"),
) -> JSONResponse:
    """跑一次离线实验：上传的音频（或内置样例）→ 加混响加噪 → 多方法对比。"""
    import io

    from rtse.audio.io import read_audio
    from rtse.server.lab import LabRequest, run_lab

    if file is not None:
        raw = await file.read()
        try:
            audio = read_audio(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                {"error": f"无法解析音频文件：{type(exc).__name__}: {exc}"}, status_code=400
            )
    else:
        audio = _demo_audio()
        if audio is None:
            return JSONResponse(
                {"error": "未上传文件，且没有内置样例。请上传一个 wav/flac/mp3。"},
                status_code=400,
            )

    result = run_lab(
        LabRequest(
            audio=audio,
            noise_kind=noise_kind,
            snr_db=snr_db,
            t60=t60,
            methods=tuple(m for m in methods.split(",") if m),
            vad=vad,
        )
    )
    return JSONResponse(result)


def _demo_audio() -> np.ndarray | None:
    """内置样例：data/demo/ 下的第一个音频文件。没有就返回 None。"""
    from rtse.audio.io import read_audio
    from rtse.paths import DATA_DIR

    demo = DATA_DIR / "demo"
    for ext in ("*.wav", "*.flac", "*.mp3"):
        for p in sorted(demo.glob(ext)):
            return read_audio(p)
    return None


@app.get("/api/results")
def api_results() -> JSONResponse:
    """基准看板的数据源：把 results/*.json 原样吐出去。"""
    out = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            out[p.stem] = {"error": f"{type(exc).__name__}: {exc}"}
    return JSONResponse({"results": out, "dir": str(RESULTS_DIR)})


# --------------------------------------------------------------------- 实时流


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    session = StreamSession(SessionConfig())
    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if (text := msg.get("text")) is not None:
                try:
                    cmd = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if cmd.get("op") == "config":
                    session.update_config(
                        **{k: v for k, v in cmd.items() if k in SessionConfig.__annotations__}
                    )
                    await ws.send_text(json.dumps({"op": "config_ack", "cfg": vars(session.cfg)}))
                elif cmd.get("op") == "reset":
                    session.rebuild()
                continue

            data = msg.get("bytes")
            if not data:
                continue

            pcm = np.frombuffer(data, dtype=np.float32).astype(np.float64)
            result = session.process(pcm)
            await ws.send_text(json.dumps(result.meta))
            await ws.send_bytes(result.audio.astype(np.float32).tobytes())

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        # 把服务端异常送到前端显示，否则表现为"界面莫名其妙不动了"
        try:
            await ws.send_text(json.dumps({"op": "error", "msg": f"{type(exc).__name__}: {exc}"}))
        except Exception:  # noqa: BLE001, S110
            pass


# --------------------------------------------------------------------- 静态页面


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/lab")
def lab() -> FileResponse:
    return FileResponse(WEB_DIR / "lab.html")


@app.get("/bench")
def bench() -> FileResponse:
    return FileResponse(WEB_DIR / "bench.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
