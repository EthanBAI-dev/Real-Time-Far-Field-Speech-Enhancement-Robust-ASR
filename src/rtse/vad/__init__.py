"""语音活动检测。三档实现互为对照，见 base.py 的说明。"""

from __future__ import annotations

from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig
from rtse.vad.base import StreamingVAD, VADFrame
from rtse.vad.energy import EnergyVAD
from rtse.vad.webrtc import WebRTCVAD

__all__ = ["StreamingVAD", "VADFrame", "EnergyVAD", "WebRTCVAD", "VAD_METHODS", "build_vad"]

VAD_METHODS: dict[str, type[StreamingVAD]] = {
    EnergyVAD.name: EnergyVAD,
    WebRTCVAD.name: WebRTCVAD,
}


def build_vad(name: str, cfg: STFTConfig = DEFAULT_CONFIG, **kwargs) -> StreamingVAD:
    if name not in VAD_METHODS:
        raise KeyError(f"未知的 VAD {name!r}，可选：{sorted(VAD_METHODS)}")
    return VAD_METHODS[name](cfg, **kwargs)
