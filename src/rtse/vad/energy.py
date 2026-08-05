"""自研能量 + 谱平坦度 VAD。

不是"能量过门限就算语音"那种玩具实现。要在真实噪声下可用，至少得解决三件事：

1. **门限必须自适应**。固定门限换个录音环境就废了。这里跟踪噪声本底，
   门限相对本底浮动。
2. **能量单靠自己不够**。稳态噪声能量也可以很高。加上**谱平坦度（SFM）**做联合判决：
   语音是谐波结构，谱起伏大、平坦度低；白噪声谱平坦度接近 1。
   两个特征的错误模式不同，联合起来鲁棒得多。
3. **必须有迟滞（hysteresis）与挂起（hangover）**。逐帧独立判决会在语音中间的
   短暂停顿处频繁切断，把一个字劈成两半。起始要连续 N 帧才确认（抗误触发），
   结束要保持 M 帧才释放（抗漏检）—— 且 M 明显大于 N，因为**漏检的代价高得多**。
"""

from __future__ import annotations

import numpy as np

from rtse import SAMPLE_RATE
from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig
from rtse.vad.base import StreamingVAD, VADFrame

__all__ = ["EnergyVAD"]

_EPS = 1e-12


class EnergyVAD(StreamingVAD):
    """能量 + 谱平坦度联合判决，带自适应本底跟踪与迟滞。

    Args:
        thresh_db: 语音带能量高出噪声本底多少 dB 才算候选语音帧。
        sfm_thresh: 谱平坦度门限，低于它才算有谐波结构。
        onset_frames: 连续多少帧候选才确认语音开始（抗误触发）。
        hangover_frames: 语音结束后再保持多少帧（抗漏检）。
            默认 12 帧 ≈ 190 ms，覆盖词间停顿和清音尾音。
        speech_band: 用于判决的频带（Hz）。限制在语音主要能量区，
            避开低频空调隆隆声和高频嘶声的干扰。
    """

    name = "energy"

    def __init__(
        self,
        cfg: STFTConfig = DEFAULT_CONFIG,
        thresh_db: float = 8.0,
        sfm_thresh: float = 0.45,
        onset_frames: int = 2,
        hangover_frames: int = 12,
        noise_adapt: float = 0.98,
        speech_band: tuple[float, float] = (250.0, 3800.0),
    ) -> None:
        super().__init__(cfg)
        self.thresh_db = thresh_db
        self.sfm_thresh = sfm_thresh
        self.onset_frames = onset_frames
        self.hangover_frames = hangover_frames
        self.noise_adapt = noise_adapt

        freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / SAMPLE_RATE)
        self._band = (freqs >= speech_band[0]) & (freqs <= speech_band[1])
        self.reset()

    def reset(self) -> None:
        self._noise_db: float | None = None
        self._onset_count = 0
        self._hang_count = 0
        self._active = False

    @staticmethod
    def _spectral_flatness(power: np.ndarray) -> float:
        """谱平坦度 = 几何平均 / 算术平均，取值 0~1。

        白噪声接近 1，纯音接近 0，浊音语音典型在 0.1~0.4。
        用对数域求几何平均，避免连乘下溢（257 个数连乘必然下溢到 0）。
        """
        p = np.maximum(power, _EPS)
        geo = np.exp(np.mean(np.log(p)))
        ari = np.mean(p)
        return float(geo / (ari + _EPS))

    def process(self, block: np.ndarray, spec: np.ndarray | None = None) -> VADFrame:
        if spec is None:
            win = self.cfg.window
            buf = np.zeros(self.cfg.n_fft)
            buf[-block.size :] = block
            spec = np.fft.rfft(buf * win)

        power = np.abs(spec) ** 2
        band_power = float(np.sum(power[self._band]))
        energy_db = 10.0 * np.log10(band_power + _EPS)
        sfm = self._spectral_flatness(power[self._band])

        # 噪声本底：只在非语音帧更新，且只允许"快降慢升"。
        # 允许快速下降是为了在噪声突然变小时及时跟上；
        # 限制上升速度是为了防止一段持续语音把本底一路抬高、最终把自己判成噪声。
        if self._noise_db is None:
            self._noise_db = energy_db
        elif not self._active:
            a = self.noise_adapt if energy_db > self._noise_db else 0.5
            self._noise_db = a * self._noise_db + (1 - a) * energy_db

        excess_db = energy_db - self._noise_db
        candidate = (excess_db > self.thresh_db) and (sfm < self.sfm_thresh)

        # 迟滞状态机
        if candidate:
            self._onset_count += 1
            if self._onset_count >= self.onset_frames:
                self._active = True
                self._hang_count = self.hangover_frames
        else:
            self._onset_count = 0
            if self._hang_count > 0:
                self._hang_count -= 1
            else:
                self._active = False

        # 概率仅用于可视化：把超出量映射到 0~1，不参与判决
        prob = float(np.clip(excess_db / (2.0 * self.thresh_db), 0.0, 1.0))
        if self._active and not candidate:
            prob = max(prob, 0.35)  # 挂起期给个中间值，界面上能看出"正在保持"
        return VADFrame(is_speech=self._active, prob=prob)
