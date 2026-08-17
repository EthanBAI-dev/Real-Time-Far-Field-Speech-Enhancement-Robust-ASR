"""``rtse-eval`` —— 在固定测试集上跑完整评测矩阵。

补上 `pyproject.toml` 里声明了入口点、但一直没有实现的那个缺口。

两类指标分开跑，因为代价差了两个数量级：

- **有参考的客观指标**（SI-SDR / SegSNR / STOI / ESTOI）：纯数值计算，
  780 样本 × 7 方法几分钟就能跑完，默认全量。
- **CER**：每条都要过一遍 Whisper，慢 100 倍以上。默认走**分层抽样**
  （每个实验格取固定条数），保证每种 SNR/噪声/T60 组合都有代表，
  而不是从头顺序取前 N 条——那样会全部落在同一个格子里。

PESQ 本机装不上（见 docs/ISSUES.md I-05），只能在 Colab 侧补，这里不算。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rtse.audio.io import read_audio
from rtse.cli._console import console

METHODS_DSP = ("specsub", "wiener", "mmse-lsa")


def _build(method: str, model_dir: Path):
    """按方法名构造增强器。``none`` 返回 None，表示不做任何处理。"""
    if method == "none":
        return None
    if method in METHODS_DSP:
        from rtse.dsp import build_dsp

        return build_dsp(method)
    from rtse.runtime import OnnxEnhancer

    return OnnxEnhancer(model_dir / f"{method}.onnx")


def _stratified(records: list[dict], per_cell: int) -> list[dict]:
    """按 (snr, noise, t60) 分组，每组取前 ``per_cell`` 条。

    直接取前 N 条会全部落在同一个实验格里（记录是按格生成的），
    那样算出来的 CER 只反映某一种噪声/SNR，没有代表性。
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["snr"], r["noise"], r["t60"])].append(r)
    out = []
    for key in sorted(buckets):
        out.extend(buckets[key][:per_cell])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("testset", nargs="?", default="data/testset", help="测试集目录，默认 data/testset")
    p.add_argument("--models", default="models", help="ONNX 模型目录，默认 models/")
    p.add_argument("--methods", default="none,specsub,wiener,mmse-lsa,crn-nano,crn-lite,crn-large")
    p.add_argument("--out", default="results/local_metrics.json")
    p.add_argument("--limit", type=int, help="只跑前 N 条（调试用）")
    p.add_argument("--skip-cer", action="store_true", help="跳过 CER（只算客观指标，快很多）")
    p.add_argument("--cer-per-cell", type=int, default=2,
                   help="CER 分层抽样：每个实验格取几条，默认 2（39 格 → 78 条）")
    p.add_argument("--asr-model", default="small", help="faster-whisper 规格，默认 small")
    p.add_argument("--restart", action="store_true",
                   help="忽略 --out 里已有的缓存，从头跑（默认是续跑）")
    args = p.parse_args()

    root = Path(args.testset)
    model_dir = Path(args.models)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    records = idx["records"]
    if args.limit:
        records = records[: args.limit]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    from rtse.metrics.intrusive import estoi, seg_snr, si_sdr, stoi
    from rtse.runtime import Pipeline
    from rtse.vad import build_vad

    console.print(f"[bold]测试集[/bold] {root}  {len(records)} 条 × {len(methods)} 方法")

    # ── 断点续跑 ──────────────────────────────────────────────────────────
    # 全量评测要跑近一小时，中途被打断（会话结束、误关终端、断电）是常态。
    # **每跑完一个方法就落盘**，重跑时跳过已完成的方法——第一次写这个脚本时
    # 只在最后统一写一次，结果任务在 5/7 处被中断，前面 50 分钟全部白跑。
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cer_rows: list[dict] = []
    if out_path.exists() and not args.restart:
        cached = json.loads(out_path.read_text(encoding="utf-8"))
        rows = cached.get("objective", [])
        cer_rows = cached.get("cer", [])
        done_obj = {r["method"] for r in rows}
        done_cer = {r["method"] for r in cer_rows}
        if done_obj or done_cer:
            console.print(f"[dim]续跑：已完成客观指标 {sorted(done_obj)}；"
                          f"已完成 CER {sorted(done_cer)}。加 --restart 可从头重跑。[/dim]")

    def save() -> None:
        out_path.write_text(
            json.dumps({"testset": str(root), "objective": rows, "cer": cer_rows},
                       ensure_ascii=False), encoding="utf-8")

    # ── 1. 客观指标（全量）────────────────────────────────────────────────
    t0 = time.time()
    done_obj = {r["method"] for r in rows}
    for mi, method in enumerate(methods):
        if method in done_obj:
            console.print(f"  [{mi + 1}/{len(methods)}] {method:<10} 已完成，跳过")
            continue
        # 增强器**每种方法只构造一次**（ONNX 会话初始化很贵，780 条重建一次
        # 就是 780 次加载），但每条样本前必须 reset() —— 增强器内部有状态
        # （噪声估计、GRU 隐状态），不清会让上一条的尾部状态泄漏到下一条开头。
        enh = _build(method, model_dir)
        pipe = Pipeline(enhancer=enh, vad=build_vad("energy"))
        for i, r in enumerate(records):
            clean = read_audio(root / r["clean"])
            noisy = read_audio(root / r["noisy"])
            pipe.reset()
            out, _ = pipe.process_signal(noisy)
            rows.append({
                "id": r["id"], "method": method, "snr": r["snr"],
                "noise": r["noise"], "t60": r["t60"],
                "si_sdr": si_sdr(clean, out), "seg_snr": seg_snr(clean, out),
                "stoi": stoi(clean, out), "estoi": estoi(clean, out),
            })
            if (i + 1) % 50 == 0 or i + 1 == len(records):
                el = time.time() - t0
                console.print(f"  [{mi + 1}/{len(methods)}] {method:<10} "
                              f"{i + 1}/{len(records)}  ({el / 60:.1f} 分钟)", end="\r")
        save()  # 每个方法跑完立刻落盘
    console.print()
    console.print(f"客观指标完成，用时 {(time.time() - t0) / 60:.1f} 分钟")

    # ── 2. CER（分层抽样）─────────────────────────────────────────────────
    # cer_rows 不重新清空——续跑时它已经从 --out 缓存里加载了上次的结果，
    # 清空的话就白跑了（第一次写这个脚本时就是这么把 50 分钟跑没的）。
    if not args.skip_cer:
        from rtse.asr.scoring import cer as cer_fn
        from rtse.asr.whisper_engine import WhisperASR

        sample = [r for r in _stratified(records, args.cer_per_cell) if r.get("text")]
        console.print(f"\n[bold]CER[/bold] 分层抽样 {len(sample)} 条 "
                      f"（每格 {args.cer_per_cell} 条）× ({len(methods)} 方法 + clean 上界)")
        engine = WhisperASR(model_size=args.asr_model)
        t1 = time.time()
        done_cer = {r["method"] for r in cer_rows}

        # clean 上界：没有它就不知道"CER 降到多少算好"
        if "clean(上界)" not in done_cer:
            for i, r in enumerate(sample):
                txt = engine.transcribe(read_audio(root / r["clean"])).text
                cer_rows.append({"id": r["id"], "method": "clean(上界)", "snr": r["snr"],
                                 "noise": r["noise"], "t60": r["t60"],
                                 "cer": cer_fn(r["text"], txt)})
                console.print(f"  clean {i + 1}/{len(sample)}", end="\r")
            save()
        else:
            console.print("  clean(上界) 已完成，跳过")

        for mi, method in enumerate(methods):
            if method in done_cer:
                console.print(f"  [{mi + 1}/{len(methods)}] {method:<10} 已完成，跳过")
                continue
            enh = _build(method, model_dir)
            pipe = Pipeline(enhancer=enh, vad=build_vad("energy"))
            for i, r in enumerate(sample):
                noisy = read_audio(root / r["noisy"])
                pipe.reset()
                out, _ = pipe.process_signal(noisy)
                txt = engine.transcribe(np.asarray(out, dtype=np.float32)).text
                cer_rows.append({"id": r["id"], "method": method, "snr": r["snr"],
                                 "noise": r["noise"], "t60": r["t60"],
                                 "cer": cer_fn(r["text"], txt)})
                el = time.time() - t1
                console.print(f"  [{mi + 1}/{len(methods)}] {method:<10} "
                              f"{i + 1}/{len(sample)}  ({el / 60:.1f} 分钟)", end="\r")
            save()  # 每个方法跑完立刻落盘
        console.print()
        console.print(f"CER 完成，用时 {(time.time() - t1) / 60:.1f} 分钟")

    # ── 3. 汇总 ───────────────────────────────────────────────────────────
    save()

    console.print(f"\n{'method':<14}{'SI-SDR':>9}{'STOI':>8}{'ESTOI':>8}{'CER':>8}")
    console.print("-" * 47)
    cer_by = defaultdict(list)
    for r in cer_rows:
        cer_by[r["method"]].append(r["cer"])
    if "clean(上界)" in cer_by:
        console.print(f"{'clean(上界)':<14}{'—':>9}{'—':>8}{'—':>8}"
                      f"{st.mean(cer_by['clean(上界)']):>8.3f}")
    for m in methods:
        g = [r for r in rows if r["method"] == m]
        c = cer_by.get(m)
        cer_s = f"{st.mean(c):.3f}" if c else "—"
        console.print(f"{m:<14}{st.mean(r['si_sdr'] for r in g):>9.2f}"
                      f"{st.mean(r['stoi'] for r in g):>8.3f}"
                      f"{st.mean(r['estoi'] for r in g):>8.3f}{cer_s:>8}")
    console.print(f"\n明细已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
