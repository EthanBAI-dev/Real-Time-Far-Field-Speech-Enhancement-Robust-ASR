"""三档 VAD 后端对比：EnergyVAD / WebRTCVAD / SileroVAD 在噪声下的漏判率。

**为什么要跑这个**：上一轮 VAD 门控实验（见 docs/ISSUES.md、`compare_vad_gate.py`）
发现打开门控让所有 DSP 方法的 CER 全面变差，根因追到 `EnergyVAD` 在噪声下把
真语音判成静音的比例高达 70%（−5dB）。当时用合成的正弦谐波"假语音"信号做诊断，
换算过 WebRTC 的漏判率接近 0%，但没跑过 Silero——**Silero 是神经网络，
在真实录音上训练的，对着合成正弦谐波信号的反应和真实语音完全不同**
（手动验证过：给它一段纯正弦谐波和一段纯噪声，它反而给噪声更高的语音概率）。
所以这次必须换成真实录音，合成假语音会给出误导性的对比结果。

**语料来源**：本机没有 V1 的 `data/testsets/*`（Colab 产物没落地到本地），
`data/testset` 也已按用户要求删除。用户下载的参考项目
`ref/speech-processing-master/Speech Enhancement/wiener/` 下有 6 段真实录音
（sf1~sf3 女声、sm1~sm3 男声，16kHz），本地就位就用它们做真语音源，
噪声用本项目自己的 `make_noise`/`mix_at_snr` 现场合成——**不拷贝、不提交
这些语音文件进本仓库**，`ref/` 是用户自己下载的独立参考项目。

用法（需要本地存在 ``ref/`` 目录）::

    uv run python scripts/compare_vad_backends.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import soundfile as sf

from rtse.audio.stft import DEFAULT_CONFIG, StreamingSTFT
from rtse.cli._console import console
from rtse.data.synth import make_noise, mix_at_snr, speech_active_mask
from rtse.vad import build_vad

REF_DIR = Path(__file__).resolve().parents[1] / "ref" / "speech-processing-master"
SPEAKERS = ("sf1", "sf2", "sf3", "sm1", "sm2", "sm3")
NOISE_KINDS = ("white", "babble", "keyboard")
SNRS = (-5, 0, 5, 10, 15)
VADS = ("energy", "webrtc", "silero")


def _find_speaker(name: str) -> Path:
    hits = list(REF_DIR.rglob(f"{name}_cln.wav"))
    if not hits:
        raise FileNotFoundError(f"ref/ 下找不到 {name}_cln.wav，检查参考项目是否就位")
    return hits[0]


def _false_negative_rate(vad_name: str, noisy: np.ndarray, true_mask: np.ndarray) -> float:
    """真语音被判成静音的比例——这是本次对比唯一关心的数字（见 base.py：
    漏检的代价远高于误触发，所以只看漏判率，不看总体准确率）。"""
    cfg = DEFAULT_CONFIG
    stft = StreamingSTFT(cfg)
    vad = build_vad(vad_name)
    vad.reset()
    n_fr = (noisy.size - cfg.n_fft) // cfg.hop
    fn = tot = 0
    for i in range(n_fr):
        block = noisy[i * cfg.hop : (i + 1) * cfg.hop]
        spec = stft.push(block)
        vf = vad.process(block, spec)
        seg = true_mask[i * cfg.hop : (i + 1) * cfg.hop]
        if seg.any():
            tot += 1
            if not vf.is_speech:
                fn += 1
    return fn / max(tot, 1) * 100.0


def main() -> int:
    if not REF_DIR.exists():
        console.print(f"[red]找不到 {REF_DIR}[/red]，这个脚本需要本地存在 ref/ 参考项目。")
        return 1

    rng = np.random.default_rng(20260819)
    speech_by_speaker = {}
    for spk in SPEAKERS:
        y, sr = sf.read(_find_speaker(spk))
        assert sr == 16000, f"{spk} 不是 16kHz：{sr}"
        speech_by_speaker[spk] = y.astype(np.float64)

    console.print(f"[bold]真实语音源[/bold]：{len(SPEAKERS)} 位说话人（来自 ref/）")
    console.print(f"[bold]噪声[/bold]：{NOISE_KINDS} × SNR {SNRS}\n")

    rows = []
    for kind in NOISE_KINDS:
        for snr in SNRS:
            per_vad = {v: [] for v in VADS}
            for spk in SPEAKERS:
                speech = speech_by_speaker[spk]
                true_mask = speech_active_mask(speech)
                noise = make_noise(kind, speech.size, rng)
                noisy, _ = mix_at_snr(speech, noise, snr, rng=rng)
                for vad_name in VADS:
                    per_vad[vad_name].append(_false_negative_rate(vad_name, noisy, true_mask))
            row = {"noise": kind, "snr": snr, **{v: st.mean(per_vad[v]) for v in VADS}}
            rows.append(row)
            console.print(
                f"{kind:<10}{snr:>5}dB   "
                + "  ".join(f"{v}={row[v]:>5.1f}%" for v in VADS)
            )

    console.print(f"\n[bold]总体平均漏判率[/bold]（{len(rows)} 组噪声/SNR 条件 × {len(SPEAKERS)} 位说话人）：")
    overall_by_vad = {}
    for v in VADS:
        overall_by_vad[v] = st.mean(r[v] for r in rows)
        console.print(f"  {v:<10} {overall_by_vad[v]:>6.1f}%")

    out_path = Path(__file__).resolve().parents[1] / "results" / "vad_backend_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "note": "本地小样本试点：真实语音来自 ref/（用户自行下载的独立参考项目，"
        "不在本仓库内，无法在其他机器复现），噪声是本项目 make_noise 现场合成的。"
        "不是 V1 正式评测，是漏判率（真语音被判成静音的比例）的定向对比。",
        "speakers": list(SPEAKERS), "noise_kinds": list(NOISE_KINDS), "snrs": list(SNRS),
        "per_condition": rows, "overall_false_negative_rate_pct": overall_by_vad,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    console.print(f"\n明细已写入 {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
