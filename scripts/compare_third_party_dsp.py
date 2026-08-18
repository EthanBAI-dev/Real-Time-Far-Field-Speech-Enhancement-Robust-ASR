"""拿第三方成熟实现做对照，判断本项目的 DSP 是不是实现得太弱。

**要回答的问题**：`rtse-eval` 显示三种 DSP 在 AISHELL 上让 CER 显著变差
（+0.030~+0.049），在 dns_objective 上 SI-SDR 收益也只有零点几 dB。
这是**我们实现得不好**，还是**单通道降噪本来就这样**？

只看自己的数字答不了这个问题，必须有外部参照系。这里选两个：

- ``logmmse``：Ephraim-Malah log-MMSE 的第三方 Python 实现，**跟我们的
  `mmse-lsa` 是同一个算法**。同算法不同实现的直接对拍——如果它明显更好，
  那就是我们写得有问题；如果差不多，说明算法本身的上限就在这里。
- ``noisereduce``：谱门限（spectral gating），实践中最常被直接抓来用的现成方案，
  代表"一个工程师不想自己写会拿什么"。

**对我们不利的一处不对称，必须说明**：这两个第三方实现都是**离线整段处理**，
可以looking ahead 用到未来帧；本项目的实现是**严格因果的流式**，
只能用当前帧和历史（这是实时约束，见 `docs/PLAN.md`）。
所以这个对照对我们是偏严格的——它们有信息优势。
如果我们在这个劣势下还能打平，说明实现没有问题。

用法（第三方包用临时依赖，不写进 pyproject）::

    uv run --with logmmse --with noisereduce python scripts/compare_third_party_dsp.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from rtse.audio.io import read_audio
from rtse.cli._console import console
from rtse.dsp import build_dsp
from rtse.metrics.intrusive import estoi, si_sdr, stoi
from rtse.runtime import Pipeline

ROOT = Path(__file__).resolve().parents[1]
TESTSET = ROOT / "data" / "testsets" / "dns_objective"
OUT = ROOT / "results" / "third_party_dsp_comparison.json"
SR = 16000


def _ours(method: str):
    """走与 rtse-eval 完全相同的管线（含 STFT/iSTFT 往返）。"""
    def run(x: np.ndarray) -> np.ndarray:
        out, _ = Pipeline(enhancer=build_dsp(method)).process_signal(x)
        return out
    return run


def _logmmse(x: np.ndarray) -> np.ndarray:
    # ⚠️ `logmmse/base.py` 在**导入时**执行 `np.seterr('raise')`，把整个进程的
    # numpy 错误策略改成全部抛异常，而且不恢复。这会连累后面跑的任何库——
    # 本脚本里 `noisereduce` 就因此崩在一个良性的 sigmoid 下溢上
    # （exp(极小负数)→0 本来完全正常）。
    # 在这里显式隔离：进来先存、出去恢复，别让第三方的全局副作用外溢。
    import logmmse

    saved = np.geterr()
    try:
        np.seterr(**{k: "ignore" for k in saved})
        xi = np.clip(x, -1.0, 1.0)
        peak = max(float(np.max(np.abs(xi))), 1e-9)
        pcm = (xi / peak * 32767.0).astype(np.int16)  # 该实现只接受 int16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = logmmse.logmmse(pcm, SR)
        return np.asarray(out, dtype=np.float64) / 32767.0 * peak
    finally:
        np.seterr(**saved)


def _noisereduce(x: np.ndarray) -> np.ndarray:
    import noisereduce as nr

    saved = np.geterr()
    try:
        np.seterr(**{k: "ignore" for k in saved})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(
                nr.reduce_noise(y=x.astype(np.float32), sr=SR, stationary=False),
                dtype=np.float64,
            )
    finally:
        np.seterr(**saved)


METHODS = [
    ("none", None, "不处理（配对基线）"),
    ("specsub", _ours("specsub"), "本项目 · 谱减法（流式因果）"),
    ("wiener", _ours("wiener"), "本项目 · 维纳（流式因果）"),
    ("mmse-lsa", _ours("mmse-lsa"), "本项目 · MMSE-LSA（流式因果）"),
    ("logmmse(第三方)", _logmmse, "第三方 · 同算法，离线整段"),
    ("noisereduce(第三方)", _noisereduce, "第三方 · 谱门限，离线整段"),
]


def _ci(vals: list[float], seed: int = 20260819, n_boot: int = 4000):
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    return float(a.mean()), *(float(v) for v in np.quantile(bs, [0.025, 0.975]))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    idx = json.loads((TESTSET / "index.json").read_text(encoding="utf-8"))
    recs = idx["records"]
    noisy = {r["id"]: read_audio(TESTSET / r["noisy"]) for r in recs}
    clean = {r["id"]: read_audio(TESTSET / r["clean"]) for r in recs}
    meta = {r["id"]: r for r in recs}
    console.print(f"[bold]{TESTSET.name}[/bold]  {len(recs)} 条  "
                  f"（其中 1 条是 clean 恒等格，汇总时单列）\n")

    results: dict[str, dict] = {}
    for name, fn, desc in METHODS:
        t0 = time.time()
        per = {}
        for rid, x in noisy.items():
            out = x if fn is None else fn(x)
            n = min(out.size, clean[rid].size)
            o, c = out[:n], clean[rid][:n]
            per[rid] = {"si_sdr": si_sdr(c, o), "stoi": stoi(c, o), "estoi": estoi(c, o)}
        results[name] = per
        console.print(f"  {name:<20} {desc:<28} {time.time() - t0:>5.1f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "testset": str(TESTSET.relative_to(ROOT)),
        "note": "第三方为离线整段处理（可用未来帧），本项目为严格因果流式——"
                "这个对照对本项目偏严格。",
        "per_sample": results,
    }, ensure_ascii=False), encoding="utf-8")

    noisy_ids = [r["id"] for r in recs if str(r["snr"]) != "clean"]
    clean_id = next(r["id"] for r in recs if str(r["snr"]) == "clean")
    base = results["none"]

    console.print(f"\n{'方法':<22}{'ΔSI-SDR':>10}{'95% CI':>19}{'ΔSTOI':>9}{'改善':>8}")
    console.print("-" * 70)
    for name, _, _ in METHODS[1:]:
        d = [results[name][i]["si_sdr"] - base[i]["si_sdr"] for i in noisy_ids]
        ds = [results[name][i]["stoi"] - base[i]["stoi"] for i in noisy_ids]
        m, lo, hi = _ci(d)
        mark = "✅" if lo > 0 else ("❌" if hi < 0 else "—")
        console.print(f"{name:<22}{m:>+10.2f}   [{lo:>+6.2f},{hi:>+6.2f}]{mark}"
                      f"{np.mean(ds):>+9.4f}{sum(1 for v in d if v > 0):>5}/{len(d)}")

    console.print(f"\nclean 恒等格（输入 {base[clean_id]['si_sdr']:.1f} dB，越高越无害；闸门 ≥20）:")
    for name, _, _ in METHODS[1:]:
        v = results[name][clean_id]["si_sdr"]
        console.print(f"  {name:<22}{v:>8.1f} dB  {'✅' if v >= 20 else '❌'}")

    console.print("\n按输入 SNR 拆开的 ΔSI-SDR:")
    snrs = ["-5", "0", "5", "10", "15"]
    console.print(f"{'方法':<22}" + "".join(f"{s + 'dB':>9}" for s in snrs))
    console.print("-" * 67)
    for name, _, _ in METHODS[1:]:
        row = f"{name:<22}"
        for s in snrs:
            sel = [i for i in noisy_ids if str(meta[i]["snr"]) == s]
            row += f"{np.mean([results[name][i]['si_sdr'] - base[i]['si_sdr'] for i in sel]):>+9.2f}"
        console.print(row)

    console.print(f"\n明细已写入 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
