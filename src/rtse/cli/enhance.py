"""``rtse-enhance`` —— 对单个音频文件做降噪，写出增强后的音频。

补上 `pyproject.toml` 里声明了入口点、但文件一直不存在的又一个缺口
（同一类问题：`rtse-eval` 也是，见 docs/ISSUES.md I-23 的排查记录）。

走的是**和实时演示完全相同的那条代码路径**（`Pipeline` 逐帧流式处理），
不是另写一份离线的向量化实现——否则这里听到的效果不等于部署时的效果。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rtse.cli._console import console

DSP_METHODS = ("specsub", "wiener", "mmse-lsa")


def main() -> int:
    p = argparse.ArgumentParser(description="对音频做降噪增强（DSP 或神经模型）")
    p.add_argument("input", help="输入音频（任意采样率/声道，内部统一转 16 kHz 单声道）")
    p.add_argument("output", help="输出 wav 路径")
    p.add_argument("--method", default="wiener",
                   help="specsub / wiener / mmse-lsa / none，或神经模型名"
                        "（如 crn-nano，会在 --models 目录下找同名 .onnx）")
    p.add_argument("--models", default="models", help="ONNX 模型目录，默认 models/")
    p.add_argument("--vad-gate", action="store_true",
                   help="打开 VAD 门控（静音段额外衰减）")
    p.add_argument("--metrics", metavar="CLEAN_WAV",
                   help="给定干净参考音频则顺带算 SI-SDR / STOI 等指标")
    args = p.parse_args()

    from rtse.audio.io import read_audio, write_audio
    from rtse.runtime import Pipeline
    from rtse.vad import build_vad

    src = Path(args.input)
    if not src.exists():
        console.print(f"[red]找不到输入文件 {src}[/red]")
        return 1

    # 构造增强器
    enhancer = None
    if args.method in DSP_METHODS:
        from rtse.dsp import build_dsp

        enhancer = build_dsp(args.method)
    elif args.method != "none":
        from rtse.runtime import OnnxEnhancer

        onnx = Path(args.models) / f"{args.method}.onnx"
        if not onnx.exists():
            available = sorted(q.stem for q in Path(args.models).glob("*.onnx"))
            console.print(f"[red]找不到模型 {onnx}[/red]")
            console.print(f"可选：{DSP_METHODS + tuple(available)}")
            return 1
        enhancer = OnnxEnhancer(onnx)

    noisy = read_audio(src)
    pipe = Pipeline(enhancer=enhancer, vad=build_vad("energy"), vad_gate=args.vad_gate)

    t0 = time.perf_counter()
    out, _ = pipe.process_signal(noisy)
    elapsed = time.perf_counter() - t0

    gain = write_audio(args.output, out)
    dur = noisy.size / 16000
    console.print(f"[bold]{args.method}[/bold]  {dur:.2f}s 音频  "
                  f"处理耗时 {elapsed * 1000:.0f} ms  RTF={elapsed / dur:.4f}")
    if gain != 1.0:
        console.print(f"[dim]输出峰值超过满幅，已整体缩放 ×{gain:.3f} 防削波[/dim]")
    console.print(f"已写入 {args.output}")

    if args.metrics:
        # 指标在**内存数组**上算，不读回刚写的文件——写盘时的削波保护会改变
        # 绝对幅度，污染 SI-SDR/SegSNR 这类非尺度不变的指标（见 ISSUES.md I-07）。
        from rtse.metrics.intrusive import estoi, seg_snr, si_sdr, stoi

        clean = read_audio(args.metrics)
        n = min(clean.size, out.size, noisy.size)
        clean, out_, noisy_ = clean[:n], out[:n], noisy[:n]
        console.print()
        console.print(f"{'':<10}{'SI-SDR':>9}{'SegSNR':>9}{'STOI':>8}{'ESTOI':>8}")
        for tag, sig in (("处理前", noisy_), ("处理后", out_)):
            console.print(f"{tag:<10}{si_sdr(clean, sig):>9.2f}{seg_snr(clean, sig):>9.2f}"
                          f"{stoi(clean, sig):>8.3f}{estoi(clean, sig):>8.3f}")
        console.print(f"{'Δ':<10}{si_sdr(clean, out_) - si_sdr(clean, noisy_):>+9.2f}"
                      f"{seg_snr(clean, out_) - seg_snr(clean, noisy_):>+9.2f}"
                      f"{stoi(clean, out_) - stoi(clean, noisy_):>+8.3f}"
                      f"{estoi(clean, out_) - estoi(clean, noisy_):>+8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
