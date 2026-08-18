"""Silero VAD 封装：神经网络方案，当前轻量方案的效果上界。

**帧长不兼容问题，和 WebRTC 那道坎是同一类，但更容易踩对**：

Silero 只接受 **固定 512 样本**（16kHz 下 32ms）的输入块，本项目的帧移是
256 样本（16ms）。跟 `webrtc.py` 一样，不改整个项目的帧移去迁就一个 VAD，
而是在这一层做适配：内部缓冲累积到 512 样本才真正跑一次模型，中间那次
`process()` 调用沿用上一次的判决——**512 正好是我们 hop 的整数倍**，
不像 WebRTC 的 160 那样跟 256 有余数，缓冲逻辑不需要处理"凑不满一个子帧"
的边界情况。

**判决口径参照 Silero 官方 `VADIterator` 的写法**：起始阈值 0.5、
结束阈值 0.5-0.15=0.35（留一段"不确定区"，避免概率在阈值附近抖动时
反复开关）；`hangover_frames` 沿用本项目其余 VAD 的单位（本地 hop 帧数，
不是 Silero 自己的毫秒参数），保持三档 VAD 对外接口一致。

模型走 ONNX（`load_silero_vad(onnx=True)`），权重打包在 `silero-vad` 这个
pip 包里，不需要联网下载；`OnnxWrapper` 内部已经把 onnxruntime 的
`inter_op/intra_op_num_threads` 锁到 1、并强制 CPU provider——
跟本项目"RTF 必须在 CPU 单线程下测"的约定天然一致，不用我们再配置。
"""

from __future__ import annotations

import numpy as np

from rtse import SAMPLE_RATE
from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig
from rtse.vad.base import StreamingVAD, VADFrame

__all__ = ["SileroVAD", "SILERO_CHUNK"]

#: Silero 模型要求的固定输入长度（16kHz 下 32ms）。不是本项目的 hop，
#: 是 Silero 训练时定死的，改不了。
SILERO_CHUNK = 512


class SileroVAD(StreamingVAD):
    """Silero 神经网络 VAD（ONNX 推理）。

    Args:
        threshold: 起始判决阈值，Silero 官方推荐 0.5（"lazy 0.5 对多数数据集
            够用"，见其 `VADIterator` 文档）。
        release_margin: 结束判决比起始阈值低多少。取 `threshold - release_margin`
            作为"确认非语音"的下限，中间这段视为不确定区，不触发也不清空挂起——
            跟 Silero 官方 `VADIterator` 里 `threshold - 0.15` 的用法一致。
        hangover_frames: 语音结束后再保持多少个**本地 hop**（不是 Silero 的
            512 样本块）。默认 12 帧 ≈ 190ms，和 `EnergyVAD` 默认值对齐，
            方便三档 VAD 在同样的挂起时长下比较判决质量本身的差异。
    """

    name = "silero"

    def __init__(
        self,
        cfg: STFTConfig = DEFAULT_CONFIG,
        threshold: float = 0.5,
        release_margin: float = 0.15,
        hangover_frames: int = 12,
    ) -> None:
        super().__init__(cfg)
        if cfg.hop <= 0 or SILERO_CHUNK % cfg.hop != 0:
            # 目前项目默认 hop=256，SILERO_CHUNK=512，整除。万一将来改了 hop，
            # 缓冲逻辑的"每 N 次 process() 触发一次推理"这个假设就不成立了，
            # 必须在这里就炸出来，而不是悄悄退化成判决对不上帧边界。
            raise ValueError(
                f"SileroVAD 要求 hop（{cfg.hop}）能整除 {SILERO_CHUNK}，否则缓冲逻辑不成立"
            )
        self.threshold = threshold
        self.release_threshold = threshold - release_margin
        self.hangover_frames = hangover_frames

        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad(onnx=True)
        self.reset()

    def reset(self) -> None:
        self._model.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
        self._active = False
        self._prob = 0.0
        self._hang_count = 0

    def process(self, block: np.ndarray, spec: np.ndarray | None = None) -> VADFrame:
        self._buf = np.concatenate([self._buf, np.asarray(block, dtype=np.float32).reshape(-1)])

        while self._buf.size >= SILERO_CHUNK:
            chunk, self._buf = self._buf[:SILERO_CHUNK], self._buf[SILERO_CHUNK:]
            x = self._torch.from_numpy(chunk)
            self._prob = float(self._model(x, SAMPLE_RATE).item())

            if self._prob >= self.threshold:
                self._active = True
                self._hang_count = self.hangover_frames
            elif self._prob < self.release_threshold:
                if self._hang_count > 0:
                    self._hang_count -= 1
                else:
                    self._active = False
            # 处于 [release_threshold, threshold) 的不确定区：既不新触发也不清空挂起，
            # 保持上一次判决——这正是留这段区间的意义，避免概率抖动导致判决反复横跳。

        return VADFrame(is_speech=self._active, prob=self._prob)
