"""VAD 统一接口。

**为什么 VAD 在这条链路里是一等公民而不是配角**：
它的错误是**不可恢复**的。降噪做差了，ASR 还有机会靠语言模型纠回来；
VAD 把一个字的起始切掉了，那个字就永远没了 —— 直接变成 CER 里的删除错误。
所以 VAD 的评测重点不是"准确率"，而是**漏检（miss）的代价远高于误触发（false alarm）**。

三档实现互为对照：
- ``EnergyVAD``：自研，纯 DSP，展示信号处理功底，可解释、可调、零依赖；
- ``WebRTCVAD``：工业界事实基线，GMM 分类器；
- ``SileroVAD``：神经网络，当前轻量方案的效果上界（需下载 ONNX）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig

__all__ = ["VADFrame", "StreamingVAD"]


@dataclass
class VADFrame:
    """单帧 VAD 判决。"""

    is_speech: bool
    prob: float
    """0~1 的语音概率。硬判决的实现（如 WebRTC）只会给出 0.0 / 1.0。"""


class StreamingVAD(ABC):
    """逐帧 VAD 基类。

    ``process`` 同时接收时域块和频谱，是因为两类实现的需求不同：
    WebRTC / Silero 要时域采样，能量类 VAD 要频谱。管线里频谱**只算一次**，
    两边共用，不重复做 FFT。
    """

    name: str = "base"

    def __init__(self, cfg: STFTConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg

    @abstractmethod
    def process(self, block: np.ndarray, spec: np.ndarray | None = None) -> VADFrame:
        """输入一个 ``hop`` 长的时域块（及可选的对应频谱），返回该帧判决。"""

    def reset(self) -> None:
        """清空状态。"""

    def process_signal(self, x: np.ndarray) -> np.ndarray:
        """整段处理，返回逐帧布尔数组。内部走流式，保证与实时行为一致。"""
        from rtse.audio.stft import StreamingSTFT, num_frames

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        n_fr = num_frames(x.size, self.cfg)
        feed = np.zeros(n_fr * self.cfg.hop)
        feed[: x.size] = x

        self.reset()
        sa = StreamingSTFT(self.cfg)
        out = np.empty(n_fr, dtype=bool)
        for i in range(n_fr):
            blk = feed[i * self.cfg.hop : (i + 1) * self.cfg.hop]
            out[i] = self.process(blk, sa.push(blk)).is_speech
        return out
