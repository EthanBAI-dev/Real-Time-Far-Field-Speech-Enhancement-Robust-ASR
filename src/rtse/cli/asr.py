"""``rtse-asr`` —— 转写单个文件，或者对测试集批量算增强前后的 CER 对比。

这是之前 `pyproject.toml` 里声明了入口点、但文件一直不存在的那个缺口
（见 docs/ISSUES.md 的 plan-vs-actual 对照）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rtse.cli._console import console


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from rtse.asr.whisper_engine import WhisperASR
    from rtse.audio.io import read_audio

    console.print(f"[dim]加载 {args.model} 模型…（首次运行会自动下载）[/dim]")
    engine = WhisperASR(model_size=args.model, language=args.language)
    audio = read_audio(args.audio)
    result = engine.transcribe(audio)
    console.print(f"[bold]识别结果[/bold]（语言 {result.language}，"
                  f"置信度 {result.language_probability:.2f}）：")
    console.print(result.text)
    if args.ref:
        from rtse.asr.scoring import cer

        c = cer(args.ref, result.text)
        console.print(f"\nCER = [bold]{c:.3f}[/bold]（相对参考文本：{args.ref}）")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """对一个测试集（`index.json` + 音频）批量转写，输出增强前后的 CER 对比。

    每条记录必须有 `noisy`（待处理输入），`clean` 是可选字段。只有测试集显式
    声明 `cer_upper_is_meaningful=true` 时才转写 clean 参考；真实会议录音本身含噪，
    不能伪装成“干净上界”。

    如果传了 `--enhanced-dir`，会额外读取该目录下同名文件（增强后的音频）
    算第三组 CER，三组放在一起才能回答"降噪到底让 ASR 好了多少"这个问题。
    """
    from rtse.asr.scoring import cer
    from rtse.asr.whisper_engine import WhisperASR
    from rtse.audio.io import read_audio

    idx = json.loads(Path(args.index).read_text(encoding="utf-8"))
    records = [r for r in idx["records"] if r.get("text")]
    if args.limit:
        records = records[: args.limit]
    if not records:
        console.print("[red]测试集里没有带转写文本的样本，无法算 CER。[/red]")
        return 1

    console.print(f"[dim]加载 {args.model} 模型…（首次运行会自动下载）[/dim]")
    engine = WhisperASR(model_size=args.model, language=args.language)
    root = Path(args.index).parent

    rows = []
    for i, r in enumerate(records):
        row = {"id": r["id"]}
        sources = [("input", "noisy")]
        if idx.get("cer_upper_is_meaningful", idx.get("reference_is_clean", True)):
            if "clean" not in r:
                raise ValueError("数据集声明存在 clean CER 上界，但记录缺少 clean 字段")
            sources.insert(0, ("clean", "clean"))
        for tag, key in sources:
            audio = read_audio(root / r[key])
            text = engine.transcribe(audio).text
            row[f"{tag}_cer"] = cer(r["text"], text)
        if args.enhanced_dir:
            enh_path = Path(args.enhanced_dir) / Path(r["noisy"]).name
            if enh_path.exists():
                text = engine.transcribe(read_audio(enh_path)).text
                row["enhanced_cer"] = cer(r["text"], text)
        rows.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            console.print(f"  {i + 1}/{len(records)}", end="\r")

    console.print()
    import statistics as st

    for key in ["clean_cer", "input_cer", "enhanced_cer"]:
        vals = [r[key] for r in rows if key in r]
        if vals:
            console.print(f"{key:<14} 均值 CER = {st.mean(vals):.3f}  (n={len(vals)})")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        console.print(f"\n明细已写入 {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="ASR 转写与 CER 评测（faster-whisper，预训练推理，不训练）")
    p.add_argument("--model", default="small", help="faster-whisper 模型规格，默认 small")
    p.add_argument("--language", default="zh", help="语言代码，默认 zh")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="转写单个音频文件")
    t.add_argument("audio", help="音频文件路径")
    t.add_argument("--ref", help="参考文本，给了就顺带算 CER")
    t.set_defaults(func=_cmd_transcribe)

    e = sub.add_parser("eval", help="对测试集批量算增强前后的 CER")
    e.add_argument("index", help="测试集 index.json 路径")
    e.add_argument("--enhanced-dir", help="增强后音频所在目录（文件名需与 noisy 音频同名）")
    e.add_argument("--limit", type=int, help="只跑前 N 条（调试用）")
    e.add_argument("--out", help="把逐条明细写到这个 json 文件")
    e.set_defaults(func=_cmd_eval)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
