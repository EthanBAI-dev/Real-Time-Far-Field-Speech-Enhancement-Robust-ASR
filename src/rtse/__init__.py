"""RTSE — 单通道实时语音降噪与中文 ASR 鲁棒性系统。"""

__version__ = "0.1.0"

# 全局信号约定。整个项目的每一个模块都必须遵守这一组常量，
# 任何一处不一致都会让 STFT 完美重构、流式一致性、延迟计量三项校验失效。
SAMPLE_RATE = 16_000
N_FFT = 512        # 32 ms 分析窗
HOP_LENGTH = 256   # 16 ms 帧移，50% 重叠
N_FREQ = N_FFT // 2 + 1  # 257

__all__ = ["HOP_LENGTH", "N_FFT", "N_FREQ", "SAMPLE_RATE", "__version__"]
