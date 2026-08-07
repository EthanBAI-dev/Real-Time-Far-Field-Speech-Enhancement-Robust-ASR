"""同一批原始语音，在旧（硬截 6 秒）和新（自然时长）测试集上各跑一次 ASR，比 CER。

这是对 I-21 修复的**决定性**检验，也是整个调查里唯一真正有效的验证手段：
前后三版基于 VAD 的启发式检测全部被证伪（假阳性率 54%~61%，见 I-21），
而"直接跑下游任务看结果"一次就给出了无歧义的答案。

2026-08-07 实测结果（small 模型，8 组高 SNR 同源样本）：
旧测试集平均 CER 0.519 → 新测试集 0.140，降低 73%，且逐条可见旧数据的转写
在句子中间戛然而止、新数据完整念完。

用法（需要本地同时存在新旧两份测试集）::

    uv run python scripts/compare_cer_old_vs_new_testset.py
"""
import json
import statistics
from pathlib import Path

from rtse.asr.scoring import cer
from rtse.asr.whisper_engine import WhisperASR
from rtse.audio.io import read_audio

NEW = Path("data/testset")
OLD = Path("data/testset_OLD_truncated_6s")

new_idx = json.loads((NEW / "index.json").read_text(encoding="utf-8"))["records"]
old_idx = json.loads((OLD / "index.json").read_text(encoding="utf-8"))["records"]

# 按 source（原始 THCHS-30 文件）配对，只取高 SNR + 主网格，减少噪声干扰
old_by_src = {r["source"]: r for r in old_idx if r["snr"] >= 15 and r["t60"] == 0.3}
pairs = []
for r in new_idx:
    if r["snr"] >= 15 and r["t60"] == 0.3 and r["source"] in old_by_src:
        pairs.append((r, old_by_src[r["source"]]))
    if len(pairs) >= 8:
        break

print(f"配对到 {len(pairs)} 组同源样本\n")
engine = WhisperASR(model_size="small")

rows = []
for new_r, old_r in pairs:
    ref = new_r["text"]
    assert ref == old_r["text"], "同源样本的参考文本应当一致"
    new_wav = read_audio(NEW / new_r["clean"])
    old_wav = read_audio(OLD / old_r["clean"])
    new_txt = engine.transcribe(new_wav).text
    old_txt = engine.transcribe(old_wav).text
    c_new, c_old = cer(ref, new_txt), cer(ref, old_txt)
    rows.append((c_old, c_new))
    print(f"参考({len(ref)}字): {ref}")
    print(f"  旧({old_wav.size / 16000:.2f}s) CER={c_old:.3f}: {old_txt}")
    print(f"  新({new_wav.size / 16000:.2f}s) CER={c_new:.3f}: {new_txt}")
    print()

print("=" * 60)
print(f"旧测试集（硬截 6 秒）平均 CER = {statistics.mean(r[0] for r in rows):.3f}")
print(f"新测试集（自然时长）平均 CER = {statistics.mean(r[1] for r in rows):.3f}")
