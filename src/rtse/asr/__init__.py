"""ASR 集成：faster-whisper 封装、中文文本归一化、CER/WER 计算。

**这里没有任何训练代码，也不需要。** faster-whisper 是 OpenAI Whisper 的推理加速版
（CTranslate2 后端），Whisper 本身已经在海量多语种数据上训练好，中文识别能力现成可用。

本项目要回答的问题是"降噪前端能不能让 ASR 的字错率降低"——这需要一个**固定不变、
本身足够强**的 ASR 系统去测量前后差异。如果自己在小规模语料上训练/微调 ASR，
测出来的"CER 变化"就分不清是降噪起了作用，还是我们自己的 ASR 本身不稳定——
用一个强的、外部固定的预训练模型，才能干净地把降噪的真实贡献单独测出来。
"""

from rtse.asr.normalize import chinese_numeral_to_arabic, normalize_cer_text, normalize_wer_text
from rtse.asr.whisper_engine import WhisperASR
from rtse.asr.scoring import cer, wer

__all__ = [
    "WhisperASR",
    "normalize_cer_text",
    "normalize_wer_text",
    "chinese_numeral_to_arabic",
    "cer",
    "wer",
]
