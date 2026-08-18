"""``rtse-eval`` —— 在固定测试集上跑完整评测矩阵。

补上 `pyproject.toml` 里声明了入口点、但一直没有实现的那个缺口。

两类指标分开跑，因为代价差了两个数量级，而且**不是每套测试集都适合两类指标**：

- **有参考的客观指标**（SI-SDR / SegSNR / STOI / ESTOI）：只有数据集声明
  `reference_is_clean=true` 时才算，默认全量。
- **CER**：每条都要过一遍 Whisper，慢 100 倍以上。默认走**分层抽样**
  （分层字段由 `index.json:strata` 声明），保证每种条件都有代表，
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


#: 受控测试集的标准分层维度。每套V1数据仍必须在 ``index.json`` 显式声明，
#: 这里仅作为低层辅助函数和单元测试的默认参数。
STRATA = ("snr", "noise_kind", "rir_kind")


def _dataset_strata(idx: dict) -> tuple[str, ...]:
    """读取数据集自己声明的分层字段；V1禁止猜测或兼容旧索引。"""
    raw = idx.get("strata")
    if raw is None:
        raise ValueError("V1 index.json 必须显式声明 strata，旧单测试集不再兼容")
    if not isinstance(raw, list) or not raw or not all(isinstance(k, str) and k for k in raw):
        raise ValueError("index.json 的 strata 必须是非空字符串列表")
    if len(set(raw)) != len(raw):
        raise ValueError(f"index.json 的 strata 有重复字段：{raw}")
    return tuple(raw)


def _cell(r: dict, strata: tuple[str, ...] = STRATA) -> tuple:
    """一条记录所属的实验格。字段缺失直接报错而不是给默认值——
    默认值会让schema 不匹配悄悄退化成"所有样本挤在同一格"，
    抽出来的 CER 只反映某一种条件却看不出异常。"""
    missing = [k for k in strata if k not in r]
    if missing:
        raise KeyError(
            f"测试集记录缺少字段 {missing}。当前 index.json 的字段是 "
            f"{sorted(r)}——这套评测按 {strata} 分层，"
            f"数据集换过之后要同步更新（见 docs/ISSUES.md I-29）。")
    return tuple(r[k] for k in strata)


def _stratified(
    records: list[dict], per_cell: int, strata: tuple[str, ...] = STRATA,
) -> list[dict]:
    """按实验格分组，每组取前 ``per_cell`` 条。

    直接取前 N 条会全部落在同一个实验格里（记录是按格生成的），
    那样算出来的 CER 只反映某一种噪声/SNR，没有代表性。
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[_cell(r, strata)].append(r)
    out = []
    # JSON 字段可能混有 None/数字/字符串，直接 tuple 排序在 Python 3 会 TypeError。
    for key in sorted(buckets, key=lambda x: tuple(str(v) for v in x)):
        out.extend(buckets[key][:per_cell])
    return out


def _cer_upper_enabled(idx: dict, records: list[dict]) -> bool:
    """clean 音频是否能代表 ASR 的受控上界。

    WenetSpeech meeting 的原始录音本身含噪与混响，不能叫 clean 上界；AISHELL
    受控集的无加性噪声参考才可以。让数据集显式声明，避免靠目录名猜。
    """
    enabled = bool(idx.get("cer_upper_is_meaningful", idx.get("reference_is_clean", True)))
    if enabled and any("clean" not in r for r in records):
        raise ValueError("数据集声明 cer_upper_is_meaningful=true，但有记录缺少 clean 字段")
    return enabled


def _mean_ci95(values: list[float], *, seed: int = 20260818, n_boot: int = 2000) -> dict:
    """确定性 bootstrap 均值95%置信区间。空输入返回 n=0。"""
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    if a.size == 1:
        value = float(a[0])
        return {"n": 1, "mean": value, "ci95": [value, value]}
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"n": int(a.size), "mean": float(a.mean()), "ci95": [float(lo), float(hi)]}


def _result_summary(rows: list[dict], cer_rows: list[dict]) -> dict:
    """按方法汇总均值、CI，并给出相对 none 的逐样本配对CER差。"""
    objective: dict[str, dict] = {}
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        objective[method] = {
            key: _mean_ci95([float(r[key]) for r in group if r.get(key) is not None])
            for key in ("si_sdr", "seg_snr", "stoi", "estoi")
        }

    cer_summary: dict[str, dict] = {}
    baseline = {r["id"]: float(r["cer"]) for r in cer_rows if r["method"] == "none"}
    for method in sorted({r["method"] for r in cer_rows}):
        group = [r for r in cer_rows if r["method"] == method]
        item = _mean_ci95([float(r["cer"]) for r in group])
        if method not in ("none", "clean(上界)") and baseline:
            deltas = [float(r["cer"]) - baseline[r["id"]] for r in group if r["id"] in baseline]
            item["delta_vs_none"] = _mean_ci95(deltas, seed=20260819)
        cer_summary[method] = item
    return {"objective": objective, "cer": cer_summary}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("testset", nargs="?", default="data/testsets/aishell_controlled",
                   help="测试集目录，默认 data/testsets/aishell_controlled")
    p.add_argument("--models", default="models", help="ONNX 模型目录，默认 models/")
    p.add_argument("--methods", default=None,
                   help="逗号分隔。默认 = none + 三种 DSP + **--models 目录下实际存在的**"
                        " *.onnx。不写死清单：硬编码的模型名和磁盘内容一旦不一致，"
                        "评测会在跑到那个方法时才崩（和 I-29 同类问题）")
    p.add_argument("--out", default="results/aishell_controlled_eval.json")
    p.add_argument("--limit", type=int, help="只跑前 N 条（调试用）")
    p.add_argument("--skip-cer", action="store_true", help="跳过 CER（只算客观指标，快很多）")
    p.add_argument("--skip-objective", action="store_true",
                   help="跳过有参考指标（SI-SDR/STOI/ESTOI）。测试集在 index.json 里"
                        "声明 reference_is_clean=false 时会自动跳过，无需手动指定")
    p.add_argument("--cer-per-cell", type=int,
                   help="CER 分层抽样每格条数；默认读取 index.json 的 cer_per_cell_default")
    p.add_argument("--asr-model", default="small", help="faster-whisper 规格，默认 small")
    p.add_argument("--restart", action="store_true",
                   help="忽略 --out 里已有的缓存，从头跑（默认是续跑）")
    args = p.parse_args()

    root = Path(args.testset)
    if not (root / "index.json").exists():
        console.print(f"[red]找不到V1测试集索引：{root / 'index.json'}[/red]")
        console.print("请从 rtse_handoff.zip 恢复 data/testsets/，不再回退到旧 data/testset。")
        return 1
    model_dir = Path(args.models)
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    records = idx["records"]
    strata = _dataset_strata(idx)
    cer_per_cell = args.cer_per_cell or int(idx.get("cer_per_cell_default", 2))
    if cer_per_cell <= 0:
        p.error("--cer-per-cell 必须大于 0")
    if args.limit:
        records = records[: args.limit]
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        # 自动发现只扫顶层 *.onnx；dnsmos 与 _untrained 等辅助目录刻意不扫。
        found = sorted(q.stem for q in model_dir.glob("*.onnx"))
        methods = ["none", *METHODS_DSP, *found]
        console.print(f"[dim]未指定 --methods，自动发现：{found or '（无 ONNX 模型）'}[/dim]")

    missing = [m for m in methods
               if m not in ("none",) + METHODS_DSP and not (model_dir / f"{m}.onnx").exists()]
    if missing:
        console.print(f"[red]找不到模型：{missing}[/red]")
        console.print(f"{model_dir}/ 下现有：{sorted(q.stem for q in model_dir.glob('*.onnx'))}")
        return 1

    from rtse.metrics.intrusive import estoi, seg_snr, si_sdr, stoi
    from rtse.runtime import Pipeline
    from rtse.vad import build_vad

    console.print(f"[bold]测试集[/bold] {root}  {len(records)} 条 × {len(methods)} 方法")
    console.print(f"[dim]用途={idx.get('purpose', 'unknown')}  分层={strata}[/dim]")

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
        cached_root = cached.get("testset")
        if cached_root and Path(cached_root) != root:
            console.print(f"[red]输出文件属于另一套测试集：{cached_root}[/red]")
            console.print("请换一个 --out，或加 --restart 明确覆盖。")
            return 1
        if cached.get("cer"):
            old_model = cached.get("asr_model", "small")
            old_per_cell = int(cached.get("cer_per_cell", 2))
            if old_model != args.asr_model or old_per_cell != cer_per_cell:
                console.print(
                    f"[red]CER缓存配置不同：已有 model={old_model}, per_cell={old_per_cell}；"
                    f"本次 model={args.asr_model}, per_cell={cer_per_cell}。[/red]"
                )
                console.print("请换一个 --out，或加 --restart 明确重跑，不能混用两套CER。")
                return 1
        rows = cached.get("objective", [])
        cer_rows = cached.get("cer", [])
        done_obj = {r["method"] for r in rows}
        done_cer = {r["method"] for r in cer_rows}
        if done_obj or done_cer:
            console.print(f"[dim]续跑：已完成客观指标 {sorted(done_obj)}；"
                          f"已完成 CER {sorted(done_cer)}。加 --restart 可从头重跑。[/dim]")

    def save() -> None:
        out_path.write_text(
            json.dumps({
                "testset": str(root),
                "dataset_id": idx.get("dataset_id", root.name),
                "purpose": idx.get("purpose", "legacy"),
                "strata": list(strata),
                "asr_model": args.asr_model,
                "cer_per_cell": cer_per_cell,
                "objective": rows,
                "cer": cer_rows,
                "summary": _result_summary(rows, cer_rows),
            }, ensure_ascii=False), encoding="utf-8")

    # ── 1. 客观指标（全量）────────────────────────────────────────────────
    # 有参考指标**要求参考信号本身是干净的**。参考若含噪，增强器把那些噪声去掉
    # 反而会偏离参考、被判低分——数值不是"效果差"而是**无意义**。
    # 让数据自己声明这一点（index.json 的 reference_is_clean），
    # 而不是靠使用者记得某份数据集有这个限制（见 docs/ISSUES.md I-30）。
    ref_clean = idx.get("reference_is_clean", True)
    skip_obj = args.skip_objective or not ref_clean
    if not skip_obj and any("clean" not in r for r in records):
        raise ValueError("reference_is_clean=true，但有记录缺少 clean 字段")
    if skip_obj and not args.skip_objective:
        console.print("[yellow]测试集声明 reference_is_clean=false，跳过有参考指标。[/yellow]")
        if idx.get("reference_note"):
            console.print(f"[dim]  {idx['reference_note']}[/dim]")

    t0 = time.time()
    done_obj = {r["method"] for r in rows}
    for mi, method in enumerate(methods):
        if skip_obj:
            break
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
                "id": r["id"], "method": method,
                **{k: r[k] for k in strata},
                "rt60_measured": r.get("rt60_measured"),
                "snr_measured": r.get("snr_measured"),
                "si_sdr": si_sdr(clean, out), "seg_snr": seg_snr(clean, out),
                "stoi": stoi(clean, out), "estoi": estoi(clean, out),
            })
            if (i + 1) % 50 == 0 or i + 1 == len(records):
                el = time.time() - t0
                console.print(f"  [{mi + 1}/{len(methods)}] {method:<10} "
                              f"{i + 1}/{len(records)}  ({el / 60:.1f} 分钟)", end="\r")
        save()  # 每个方法跑完立刻落盘
    console.print()
    if not skip_obj:
        console.print(f"客观指标完成，用时 {(time.time() - t0) / 60:.1f} 分钟")

    # ── 2. CER（分层抽样）─────────────────────────────────────────────────
    # cer_rows 不重新清空——续跑时它已经从 --out 缓存里加载了上次的结果，
    # 清空的话就白跑了（第一次写这个脚本时就是这么把 50 分钟跑没的）。
    if not args.skip_cer:
        from rtse.asr.scoring import cer as cer_fn
        from rtse.asr.whisper_engine import WhisperASR

        text_records = [r for r in records if r.get("text")]
        sample = _stratified(text_records, cer_per_cell, strata) if text_records else []
        if not sample:
            console.print("\n[dim]测试集没有转写文本，跳过 CER。[/dim]")
        else:
            clean_upper = _cer_upper_enabled(idx, sample)
            suffix = " + clean 上界" if clean_upper else "（无 clean 上界）"
            console.print(f"\n[bold]CER[/bold] 分层抽样 {len(sample)} 条 "
                          f"（每格 {cer_per_cell} 条）× ({len(methods)} 方法{suffix})")
            engine = WhisperASR(model_size=args.asr_model)
            t1 = time.time()
            done_cer = {r["method"] for r in cer_rows}

        # 只有 AISHELL 这类受控集才有 clean 上界。真实会议录音禁止伪装成 clean。
        if sample and clean_upper and "clean(上界)" not in done_cer:
            for i, r in enumerate(sample):
                txt = engine.transcribe(read_audio(root / r["clean"])).text
                cer_rows.append({"id": r["id"], "method": "clean(上界)",
                                 **{k: r[k] for k in strata},
                                 "cer": cer_fn(r["text"], txt)})
                console.print(f"  clean {i + 1}/{len(sample)}", end="\r")
            save()
        elif sample and clean_upper:
            console.print("  clean(上界) 已完成，跳过")

        for mi, method in enumerate(methods) if sample else []:
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
                cer_rows.append({"id": r["id"], "method": method,
                                 **{k: r[k] for k in strata},
                                 "cer": cer_fn(r["text"], txt)})
                el = time.time() - t1
                console.print(f"  [{mi + 1}/{len(methods)}] {method:<10} "
                              f"{i + 1}/{len(sample)}  ({el / 60:.1f} 分钟)", end="\r")
            save()  # 每个方法跑完立刻落盘
        if sample:
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
        if g:
            obj = (f"{st.mean(r['si_sdr'] for r in g):>9.2f}"
                   f"{st.mean(r['stoi'] for r in g):>8.3f}"
                   f"{st.mean(r['estoi'] for r in g):>8.3f}")
        else:
            obj = f"{'n/a':>9}{'n/a':>8}{'n/a':>8}"
        console.print(f"{m:<14}{obj}{cer_s:>8}")
    console.print(f"\n明细已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
