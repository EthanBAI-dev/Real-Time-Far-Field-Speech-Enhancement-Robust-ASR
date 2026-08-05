"""WebRTC VAD 封装。

**这里有一个必须处理的帧长不兼容问题**：

WebRTC VAD 只接受 **10 / 20 / 30 ms** 的 16-bit PCM 帧，即 16 kHz 下的
160 / 320 / 480 个样本。而本项目的帧移是 **256 样本（16 ms）** —— 不在允许列表里。
直接把 256 个样本喂进去会抛 ``Error while processing frame``。

这类"参数看起来差不多、实际不兼容"的接缝，是集成第三方组件时最常见的坑。
解法不是把整个项目的帧移改成 10 ms（那会连带改变 STFT 分辨率、模型结构、
延迟预算 —— 为了迁就一个基线 VAD 去动地基是本末倒置），而是在这一层做适配：

内部维持一个样本缓冲，每凑够 160 样本（10 ms）就调一次 WebRTC，
再把这些子判决**归约**成本帧的判决。归约用 **any（有一个算一个）** 而不是 majority：
漏检的代价远高于误触发（见 ``base.py`` 的说明）。

副作用是判决相对本帧最多滞后 96 个样本（6 ms），这个延迟会记进指标表。
"""

from __future__ import annotations

import numpy as np

from rtse import SAMPLE_RATE
from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig
from rtse.vad.base import StreamingVAD, VADFrame

__all__ = ["WebRTCVAD", "WEBRTC_SUBFRAME"]

#: WebRTC 允许的最小帧长（10 ms @ 16 kHz）。选最小值以尽量减小适配带来的滞后。
WEBRTC_SUBFRAME = SAMPLE_RATE // 100  # 160


class WebRTCVAD(StreamingVAD):
    """WebRTC GMM-based VAD。工业界事实基线。

    Args:
        aggressiveness: 0~3，越大越激进（越倾向判为非语音）。
            2 是常用折中；3 在低 SNR 下会开始吃掉弱语音。
    """

    name = "webrtc"

    def __init__(self, cfg: STFTConfig = DEFAULT_CONFIG, aggressiveness: int = 2) -> None:
        super().__init__(cfg)
        import webrtcvad

        if not 0 <= aggressiveness <= 3:
            raise ValueError(f"aggressiveness 必须在 0~3，收到 {aggressiveness}")
        self.aggressiveness = aggressiveness
        self._vad = webrtcvad.Vad(aggressiveness)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float64)
        self._last = False

    def process(self, block: np.ndarray, spec: np.ndarray | None = None) -> VADFrame:
        self._buf = np.concatenate([self._buf, np.asarray(block, dtype=np.float64).reshape(-1)])

        decisions: list[bool] = []
        while self._buf.size >= WEBRTC_SUBFRAME:
            chunk, self._buf = self._buf[:WEBRTC_SUBFRAME], self._buf[WEBRTC_SUBFRAME:]
            # WebRTC 要 16-bit PCM 字节流。先限幅再转换：
            # 超出 [-1,1] 的样本直接乘 32767 会整数溢出绕回，
            # 表现为随机的错误判决，且极难定位。
            pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            decisions.append(self._vad.is_speech(pcm, SAMPLE_RATE))

        if decisions:
            # any 归约：宁可误触发，不可漏检
            self._last = any(decisions)
        # 若本帧没凑够子帧（不会发生在 hop=256 > 160 的默认配置，但要对小 hop 成立），
        # 沿用上一帧判决而不是默认为静音
        return VADFrame(is_speech=self._last, prob=1.0 if self._last else 0.0)
