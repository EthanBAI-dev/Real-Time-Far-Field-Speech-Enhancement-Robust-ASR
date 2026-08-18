"""在另外两套 V1 测试集上复核"高 SNR 收手"（I-35），重点看它对 **CER** 的影响。

**为什么必须单独跑这个**：I-35 的参数 `(2,14)` 是在 `dns_objective` 的 80 条
带噪样本上扫出来的，**调参与评测用了同一批数据**，有过拟合风险；而且那套数据
只回答音质（SI-SDR/STOI），不回答下游 ASR。

M-01 反复强调的正是这一点：**SI-SDR 涨了不等于 CER 会降**。
一个只在音质指标上验证过的改动，不能默认对 ASR 也有利。

两套数据各自回答一个问题：

- ``aishell_controlled``：受控中文 CER。有无加性噪声的参考，
  所以 SI-SDR 与 CER 都成立，且有 clean 上界可以对照。
- ``wenetspeech_real``：真实会议录音的 CER。参考本身含噪含混响
  （见 I-30），**只看 CER**，有参考指标无意义。

只跑需要对比的配置（none / wiener 开关 / mmse-lsa 开关），
不重跑 specsub 和 crn-nano —— 它们不受这个改动影响。

用法::

    uv run python scripts/verify_backoff_on_cer.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from rtse.asr.scoring import cer as cer_fn
from rtse.asr.whisper_engine import WhisperASR
from rtse.audio.io import read_audio
from rtse.cli._console import console
from rtse.dsp.enhancers import BACKOFF_SNR_DB, MMSELogSTSA, WienerFilter
from rtse.metrics.intrusive import si_sdr
from rtse.runtime import Pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "backoff_cer_verification.json"
DATASETS = ("aishell_controlled", "wenetspeech_real")
ASR_MODEL = "small"

#: (标签, 构造函数)。None 表示不处理，作为配对基线。
CONFIGS: list[tuple[str, object]] = [
    ("none", None),
    ("wiener 收手关", lambda: WienerFilter(backoff_snr_db=None)),
    ("wiener 收手开", lambda: WienerFilter(backoff_snr_db=BACKOFF_SNR_DB)),
    ("mmse-lsa 收手关", lambda: MMSELogSTSA(backoff_snr_db=None)),
    ("mmse-lsa 收手开", lambda: MMSELogSTSA(backoff_snr_db=BACKOFF_SNR_DB)),
]


def _ci(values: list[float], seed: int = 20260819, n_boot: int = 4000) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return float(a.mean()), float(lo), float(hi)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    engine = WhisperASR(model_size=ASR_MODEL)
    report: dict = {
        "backoff_snr_db": list(BACKOFF_SNR_DB),
        "asr_model": ASR_MODEL,
        "note": "复核 I-35 的收手机制对 CER 的影响；参数是在 dns_objective 上扫的，"
        "这里换数据集验证是否过拟合。",
        "datasets": {},
    }

    for ds in DATASETS:
        root = ROOT / "data" / "testsets" / ds
        idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
        recs = [r for r in idx["records"] if r.get("text")]
        ref_clean = bool(idx.get("reference_is_clean", False))
        console.print(f"\n[bold]{ds}[/bold]  {len(recs)} 条  "
                      f"（参考干净={ref_clean}，{'算 SI-SDR' if ref_clean else '只算 CER'}）")

        cache = {r["id"]: read_audio(root / r["noisy"]) for r in recs}
        clean_cache = (
            {r["id"]: read_audio(root / r["clean"]) for r in recs} if ref_clean else {}
        )

        per_cfg: dict[str, dict] = {}
        t0 = time.time()
        for label, factory in CONFIGS:
            cers, sisdrs = {}, {}
            for r in recs:
                x = cache[r["id"]]
                if factory is None:
                    out = x
                else:
                    # 走与 rtse-eval 相同的管线（含 STFT/iSTFT 往返），
                    # 否则"直接调增强器"和"评测里实际发生的"会有细微差别。
                    pipe = Pipeline(enhancer=factory())
                    out, _ = pipe.process_signal(x)
                txt = engine.transcribe(np.asarray(out, dtype=np.float32)).text
                cers[r["id"]] = cer_fn(r["text"], txt)
                if ref_clean:
                    sisdrs[r["id"]] = si_sdr(clean_cache[r["id"]], out)
            per_cfg[label] = {"cer": cers, "si_sdr": sisdrs}
            console.print(f"  {label:<16} 完成  ({(time.time() - t0) / 60:.1f} 分钟)")

        report["datasets"][ds] = {
            "reference_is_clean": ref_clean,
            "n": len(recs),
            "per_config": {k: {"cer": v["cer"], "si_sdr": v["si_sdr"]} for k, v in per_cfg.items()},
        }

        # ── 汇总：CER 相对 none 的逐样本配对差（越负越好）──────────────
        base_cer = per_cfg["none"]["cer"]
        console.print(f"\n  {'配置':<16}{'CER':>8}{'ΔCER vs none':>16}{'95% CI':>20}")
        console.print("  " + "-" * 58)
        console.print(f"  {'none':<16}{np.mean(list(base_cer.values())):>8.4f}"
                      f"{'—':>16}{'—':>20}")
        for label, _ in CONFIGS[1:]:
            c = per_cfg[label]["cer"]
            d = [c[k] - base_cer[k] for k in c]
            m, lo, hi = _ci(d)
            console.print(f"  {label:<16}{np.mean(list(c.values())):>8.4f}"
                          f"{m:>+16.4f}   [{lo:>+7.4f},{hi:>+7.4f}]")

        if ref_clean:
            base_s = per_cfg["none"]["si_sdr"]
            console.print(f"\n  {'配置':<16}{'SI-SDR':>9}{'ΔSI-SDR':>12}{'95% CI':>20}")
            console.print("  " + "-" * 57)
            console.print(f"  {'none':<16}{np.mean(list(base_s.values())):>9.2f}"
                          f"{'—':>12}{'—':>20}")
            for label, _ in CONFIGS[1:]:
                s = per_cfg[label]["si_sdr"]
                d = [s[k] - base_s[k] for k in s]
                m, lo, hi = _ci(d)
                console.print(f"  {label:<16}{np.mean(list(s.values())):>9.2f}"
                              f"{m:>+12.2f}   [{lo:>+6.2f},{hi:>+6.2f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n明细已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
