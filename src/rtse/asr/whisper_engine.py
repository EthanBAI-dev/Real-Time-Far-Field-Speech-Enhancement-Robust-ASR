"""faster-whisper 封装。**只做推理，不做训练**——见 ``rtse.asr`` 模块文档说明原因。

模型权重不随本仓库分发，首次调用时由 faster-whisper 自动从 Hugging Face Hub
下载到本地缓存（`~/.cache/huggingface`），之后离线可用。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from rtse import SAMPLE_RATE

__all__ = ["WhisperASR", "Transcript", "Segment"]


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    """整段拼接文本，已去除首尾空白。"""
    segments: list[Segment]
    language: str
    language_probability: float


class WhisperASR:
    """faster-whisper 的最小封装：喂 16kHz 单声道波形，吐出文本。

    Args:
        model_size: faster-whisper 的模型规格。中文识别质量随模型增大明显提升
            （不像英文那样小模型也够用），``small`` 是本地 CPU 批量评测下
            质量/速度的合理折中；要更高精度可以换 ``medium``，代价是慢不少。
        device: ``"cpu"`` / ``"cuda"`` / ``"auto"``。评测/CER 计算这类离线批处理
            用 CPU 完全够（不像 RTF/延迟那样需要卡在单线程上量真实部署开销——
            这里只是要一个稳定的转写结果，用哪个 provider 不影响 CER 数字本身）。
        compute_type: ``int8`` 在 CPU 上最快，对转写文本的影响可以忽略
            （这是量化推理的精度，不是识别准确率的量级）。
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "zh",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None  # 惰性加载：不用 ASR 的代码路径不该为它付下载/加载的代价

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE, vad_filter: bool = False
    ) -> Transcript:
        """转写一段音频。

        Args:
            audio: 一维波形，float32/float64，幅度约在 [-1, 1]。
            sr: 采样率。不是 16000 时会报错——项目内部统一约定 16kHz，
                调用方在边界上负责重采样（``rtse.audio.io.resample``），
                这里不做隐式转换以免掩盖上游的采样率 bug。
            vad_filter: 是否用 faster-whisper 内置的 VAD 跳过静音段。
                本项目自己的评测/直播管线已经做过 VAD 分段，这里默认关闭，
                避免两层 VAD 叠加导致行为难以预测；对未预先分段的长音频
                可以打开。
        """
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"WhisperASR 只接受 {SAMPLE_RATE} Hz 输入，收到 {sr} Hz。"
                "请先用 rtse.audio.io.resample 转换。"
            )
        x = np.asarray(audio, dtype=np.float32).reshape(-1)

        segments_iter, info = self.model.transcribe(
            x, language=self.language, vad_filter=vad_filter, without_timestamps=False
        )
        segments = [Segment(s.start, s.end, s.text.strip()) for s in segments_iter]
        text = "".join(s.text for s in segments).strip()
        return Transcript(
            text=text,
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
        )


@lru_cache(maxsize=4)
def _cached_engine(model_size: str, device: str, compute_type: str, language: str) -> WhisperASR:
    """按参数缓存引擎实例，避免评测脚本里循环调用时反复重新加载模型
    （加载一次要几秒到几十秒，评测几千个样本时这个开销会被放大成显著时间）。
    """
    return WhisperASR(model_size, device, compute_type, language)


def get_engine(
    model_size: str = "small", device: str = "cpu", compute_type: str = "int8", language: str = "zh"
) -> WhisperASR:
    """获取（或复用）一个 :class:`WhisperASR` 实例。"""
    return _cached_engine(model_size, device, compute_type, language)
