"""全量校验测试集的参考音频有没有被截断（docs/ISSUES.md I-21）。

01_data_prep.ipynb 里加的校验只抽查 120 条，够用来在生成过程中快速发现系统性问题，
但不是对下载回本地的那份数据的独立复核。这个脚本对**全部**样本跑同一个 VAD
尾部检测，用来在正式拿一份新 testset.zip 去算 CER 之前做一次没有偷懒的确认。

用法::

    uv run python scripts/verify_testset_no_truncation.py data/testset
    uv run python scripts/verify_testset_no_truncation.py data/testset_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rtse import HOP_LENGTH
from rtse.audio.io import read_audio
from rtse.cli._console import console  # 本机 cp932 控制台防护，见 docs/ISSUES.md I-11
from rtse.vad import build_vad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("testset_dir", help="测试集目录，需包含 index.json")
    p.add_argument("--tail-sec", type=float, default=0.5, help="检查末尾多少秒，默认 0.5")
    p.add_argument("--out", help="把每条样本的检测结果写到这个 json 文件（可选）")
    args = p.parse_args()

    root = Path(args.testset_dir)
    index_path = root / "index.json"
    if not index_path.exists():
        console.print(f"[red]找不到 {index_path}[/red]")
        return 1

    idx = json.loads(index_path.read_text(encoding="utf-8"))
    records = idx["records"]
    sr = idx.get("sample_rate", 16000)
    # VAD 输出按 STFT 帧率算（hop=256 @16kHz = 16ms/帧），
    # 跟 01_data_prep.ipynb 里 `flags[-31:]`（约 0.5 秒）用的是同一个换算。
    tail_frames = int(round(args.tail_sec * sr / HOP_LENGTH))

    vad = build_vad("energy")
    flagged = []
    for i, r in enumerate(records):
        clean = read_audio(root / r["clean"])
        flags = vad.process_signal(clean)
        tail = flags[-tail_frames:] if tail_frames else flags[-1:]
        if tail.size and tail.mean() > 0.5:
            flagged.append(r["id"])
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            console.print(f"  {i + 1}/{len(records)}", end="\r")
    console.print()

    n = len(records)
    console.print(f"全量校验 {n} 条，疑似截断 {len(flagged)} 条 ({len(flagged) / n:.1%})")
    if flagged:
        console.print(f"前 10 条 id： {flagged[:10]}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"total": n, "flagged": flagged}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        console.print(f"明细已写入 {args.out}")

    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
