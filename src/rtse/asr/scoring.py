"""CER / WER 计算。薄封装在 jiwer 上，但归一化这一步不能省。

**WER 对中文没有意义，别拿来用**：jiwer 的 WER 是按空白切"词"算编辑距离，
中文原文没有空格，一句话会被当成**一整个词**，于是 WER 只会是 0% 或 100%，
跟真实识别质量毫无关系（实测：7 个字错 1 个，CER=14.3%，WER 却是 100%）。
中文只报 CER；WER 留给英文（LibriSpeech 跨语言对照）用，那边是空格分词的。
"""

from __future__ import annotations

import jiwer

from rtse.asr.normalize import normalize_cer_text, normalize_wer_text

__all__ = ["cer", "wer"]


def cer(reference: str, hypothesis: str) -> float:
    """字符错误率（0~1）。参考文本和识别结果都会先过同一套归一化——
    两边归一化程度不对等，测出来的差异就还是格式噪声，不是真实识别误差。
    """
    ref = normalize_cer_text(reference)
    hyp = normalize_cer_text(hypothesis)
    if not ref:
        # 空参考文本时，误差率没有良定义的分母；有输出就算全错，没输出算 0。
        return 1.0 if hyp else 0.0
    return float(jiwer.cer(ref, hyp))


def wer(reference: str, hypothesis: str) -> float:
    """词错误率（0~1），**只用于空格分词的语言**（比如英文）。

    中文文本没有空格，直接调这个函数会把整句话当成一个词，WER 只会是 0 或 1，
    没有信息量——中文场景请用 :func:`cer`。
    """
    ref = normalize_wer_text(reference)
    hyp = normalize_wer_text(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    return float(jiwer.wer(ref, hyp))
