"""全量校验测试集的参考音频有没有被截断（docs/ISSUES.md I-21）。

⚠️ **这个脚本的检测方法已被证伪，保留仅作调查记录，不要用它的结论做判断。**

2026-08-07 实测：对**已经确认修复正确**的新测试集（同源样本 CER 从 0.519 降到
0.140，逐条肉眼可见句子被念完整了），本脚本在最可信的 T60=0 那一组仍报出
12/20 "疑似截断"。也就是说 60% 的假阳性率——这个基于 VAD 尾部活动的启发式
**无法区分"被截断"和"正常录音刚好说到接近文件末尾"**（能量 VAD 的 hangover
时长与真实录音的收尾静音量级相当，见 I-21 第三次踩坑的分析）。

**真正有效的两种验证方式**（都已实施）：
1. **生成侧**：`01_data_prep.ipynb` 里的结构性断言 `assert clean.size <= seg_len`
   ——纯整数逻辑，数学上可证明，零声学假设。
2. **验收侧**：`scripts/compare_cer_old_vs_new_testset.py`
   ——直接跑 ASR 比 CER，这是唯一能真正回答"文本对不对得上音频"的检验。

**已知局限**：本地只有打包好的 `testset.zip`，里面 `clean` 字段是**混响后**的
参考音频，没有单独保存混响前的干信号。混响卷积会让语音末尾在 T60 量级上
真实地衰减拖尾——这是物理上合理的现象，不是内容被截断，但会让基于混响后
信号的 VAD 尾部检测产生假警报（首次在 Colab 上验证过：T60=0.3 的主表批量
误报，见 I-21）。所以这个脚本按 `t60` 分组报告：**T60=0 的样本**（clean 就是
干信号，没有这个问题）是可信的截断信号；**T60>0 的样本**的"疑似截断"计数
只作参考，不能直接当作数据有问题的证据。

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
    flagged_t60_0, flagged_t60_pos = [], []
    for i, r in enumerate(records):
        clean = read_audio(root / r["clean"])
        flags = vad.process_signal(clean)
        tail = flags[-tail_frames:] if tail_frames else flags[-1:]
        if tail.size and tail.mean() > 0.5:
            (flagged_t60_0 if r.get("t60", 0) == 0 else flagged_t60_pos).append(r["id"])
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            console.print(f"  {i + 1}/{len(records)}", end="\r")
    console.print()

    n = len(records)
    n_t60_0 = sum(1 for r in records if r.get("t60", 0) == 0)
    n_t60_pos = n - n_t60_0
    console.print(f"全量校验 {n} 条（T60=0 有 {n_t60_0} 条，T60>0 有 {n_t60_pos} 条）")
    console.print(
        f"[bold]T60=0（可信）[/bold]：疑似截断 {len(flagged_t60_0)} 条"
        + (f" / {n_t60_0} ({len(flagged_t60_0) / n_t60_0:.1%})" if n_t60_0 else "（无此类样本）")
    )
    if flagged_t60_0:
        console.print(f"  前 10 条 id： {flagged_t60_0[:10]}")
    console.print(
        f"[dim]T60>0（仅供参考，混响拖尾会造成假警报，不能直接当证据）[/dim]："
        f"疑似截断 {len(flagged_t60_pos)}"
        + (f" / {n_t60_pos} ({len(flagged_t60_pos) / n_t60_pos:.1%})" if n_t60_pos else "")
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "total": n,
                    "flagged_t60_0_reliable": flagged_t60_0,
                    "flagged_t60_positive_reverb_tail_caveat": flagged_t60_pos,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        console.print(f"明细已写入 {args.out}")

    return 1 if flagged_t60_0 else 0


if __name__ == "__main__":
    sys.exit(main())
