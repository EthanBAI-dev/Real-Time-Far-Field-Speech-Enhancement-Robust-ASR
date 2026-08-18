"""VAD 门控本地对比实验：DSP 增强器加不加 VAD 门控，客观指标与 CER 有没有差别。

**为什么要跑这个**：之前查代码发现，`rtse-eval` 虽然构造了 VAD，但从没把
`vad_gate=True` 传给 `Pipeline`——DSP 方法的 CER 数字，历史上全部是在 VAD
完全不参与增强的情况下跑出来的。这个脚本用现有代码给出的 `--vad-gate` 开关，
跑一次真正的对比。

**用的是哪份数据、为什么**：本机没有 V1 的 `data/testsets/*`（那是 Colab 训练用的
handoff 产物，没在本地落地），只有更早一版的 `data/testset`（`dns_wenetspeech`
schema）。这份数据后来被发现参考信号（WenetSpeech 真实会议录音）本身不干净
（见 docs/ISSUES.md I-30），所以：

- **SI-SDR/STOI 的绝对值不可信**——参考信号自带的噪声/混响会让"正确去噪"
  反而被判低分。但门控开/关用的是**同一个参考**，逐文件配对做差时这个偏差
  会被抵消掉，所以"差值"（门控开 − 门控关）仍然有信息量，只是不能拿绝对值
  当质量结论。
- **CER 不受这个问题影响**（比的是识别文本，不经过参考音频），是本次实验的
  主指标。

这是一次本地试点，不是 V1 的正式评测——正式结论仍需在 V1 数据上用 `rtse-eval
--vad-gate` 跑。用法::

    uv run python scripts/compare_vad_gate.py
"""

from __future__ import annotations

import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rtse.asr.scoring import cer as cer_fn
from rtse.asr.whisper_engine import WhisperASR
from rtse.audio.io import read_audio
from rtse.cli._console import console
from rtse.cli.evaluate import _make_pipeline
from rtse.dsp import build_dsp
from rtse.metrics.intrusive import si_sdr, stoi

TESTSET = Path("data/testset")
METHODS = ("specsub", "wiener", "mmse-lsa")
CER_PER_CELL = 2  # 每个 (noise_kind, snr) 格取 2 条
ASR_MODEL = "small"
OUT = Path("results/vad_gate_pilot.json")


def _stratified_sample(records: list[dict], per_cell: int) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["noise_kind"], r["snr"])].append(r)
    out = []
    for key in sorted(buckets):
        out.extend(buckets[key][:per_cell])
    return out


def main() -> int:
    idx = json.loads((TESTSET / "index.json").read_text(encoding="utf-8"))
    records = idx["records"]
    console.print(f"[bold]客观指标[/bold]：{len(records)} 条 × {len(METHODS)} 方法 × 2 档门控")

    # ── 1. 客观指标：全量，取门控开/关的**逐文件配对差值** ──────────────
    obj_rows = []
    t0 = time.time()
    for method in METHODS:
        pipe_off = _make_pipeline(build_dsp(method), "energy", False, -20.0)
        pipe_on = _make_pipeline(build_dsp(method), "energy", True, -20.0)
        for i, r in enumerate(records):
            clean = read_audio(TESTSET / r["clean"])
            noisy = read_audio(TESTSET / r["noisy"])
            pipe_off.reset()
            pipe_on.reset()
            out_off, _ = pipe_off.process_signal(noisy)
            out_on, _ = pipe_on.process_signal(noisy)
            obj_rows.append({
                "id": r["id"], "method": method,
                "noise_kind": r["noise_kind"], "snr": r["snr"],
                "si_sdr_off": si_sdr(clean, out_off), "si_sdr_on": si_sdr(clean, out_on),
                "stoi_off": stoi(clean, out_off), "stoi_on": stoi(clean, out_on),
            })
            if (i + 1) % 150 == 0:
                console.print(f"  {method:<10} {i + 1}/{len(records)}  "
                              f"({(time.time() - t0) / 60:.1f} 分钟)", end="\r")
    console.print(f"\n客观指标完成，用时 {(time.time() - t0) / 60:.1f} 分钟")

    # ── 2. CER：分层抽样，门控开/关 + none 基线 ──────────────────────────
    sample = _stratified_sample(records, CER_PER_CELL)
    console.print(f"\n[bold]CER[/bold]：{len(sample)} 条 × "
                  f"({len(METHODS)} 方法 + none) × 2 档门控")
    engine = WhisperASR(model_size=ASR_MODEL)
    t1 = time.time()
    cer_rows = []

    for gate in (False, True):
        pipe_none = _make_pipeline(None, "energy", gate, -20.0)
        for r in sample:
            noisy = read_audio(TESTSET / r["noisy"])
            pipe_none.reset()
            out, _ = pipe_none.process_signal(noisy)
            txt = engine.transcribe(np.asarray(out, dtype=np.float32)).text
            cer_rows.append({"id": r["id"], "method": "none", "gate": gate,
                              "noise_kind": r["noise_kind"], "snr": r["snr"],
                              "cer": cer_fn(r["text"], txt)})
        for method in METHODS:
            pipe = _make_pipeline(build_dsp(method), "energy", gate, -20.0)
            for i, r in enumerate(sample):
                noisy = read_audio(TESTSET / r["noisy"])
                pipe.reset()
                out, _ = pipe.process_signal(noisy)
                txt = engine.transcribe(np.asarray(out, dtype=np.float32)).text
                cer_rows.append({"id": r["id"], "method": method, "gate": gate,
                                  "noise_kind": r["noise_kind"], "snr": r["snr"],
                                  "cer": cer_fn(r["text"], txt)})
            console.print(f"  gate={gate!s:<5} {method:<10} 完成 "
                          f"({(time.time() - t1) / 60:.1f} 分钟)", end="\r")
    console.print(f"\nCER 完成，用时 {(time.time() - t1) / 60:.1f} 分钟")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "testset": str(TESTSET), "note": "老版 dns_wenetspeech schema，非 V1；"
        "SI-SDR/STOI 绝对值受 I-30 影响，只看门控开关的配对差值",
        "objective": obj_rows, "cer": cer_rows,
    }, ensure_ascii=False), encoding="utf-8")

    # ── 3. 汇总 ───────────────────────────────────────────────────────────
    console.print(f"\n{'method':<12}{'ΔSI-SDR(on-off)':>18}{'ΔSTOI(on-off)':>16}")
    console.print("-" * 46)
    for method in METHODS:
        g = [r for r in obj_rows if r["method"] == method]
        d_sdr = st.mean(r["si_sdr_on"] - r["si_sdr_off"] for r in g)
        d_stoi = st.mean(r["stoi_on"] - r["stoi_off"] for r in g)
        console.print(f"{method:<12}{d_sdr:>18.3f}{d_stoi:>16.4f}")

    console.print(f"\n{'method':<12}{'CER gate=off':>14}{'CER gate=on':>14}{'Δ(on-off)':>12}")
    console.print("-" * 52)
    for method in ("none", *METHODS):
        off = [r["cer"] for r in cer_rows if r["method"] == method and not r["gate"]]
        on = [r["cer"] for r in cer_rows if r["method"] == method and r["gate"]]
        console.print(f"{method:<12}{st.mean(off):>14.4f}{st.mean(on):>14.4f}"
                      f"{st.mean(on) - st.mean(off):>+12.4f}")

    console.print(f"\n明细已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
