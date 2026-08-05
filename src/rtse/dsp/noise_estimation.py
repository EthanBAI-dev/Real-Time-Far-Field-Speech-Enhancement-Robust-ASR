"""噪声功率谱估计。

**为什么不用"前 N 帧是纯噪声"这个假设**：它是教科书里最常见的简化，也是实际部署中
最先崩掉的一环。真实场景里录音一开始就可能有人在说话，而且噪声本身是非平稳的
（有人推椅子、空调启停、街上过一辆车）。一旦噪声估计错了，后面无论用谱减、维纳
还是 MMSE-LSA，都是在错误的基础上做精细计算。

这里实现 **MCRA**（Minima-Controlled Recursive Averaging, Cohen & Berdugo 2002）：
用"平滑功率谱的局部最小值"来判断每个频点当前是否有语音，
只在语音不活跃的频点上更新噪声估计。它天然支持逐帧流式，且不需要任何初始纯噪声段。
"""

from __future__ import annotations

import numpy as np

from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig

__all__ = ["MCRANoiseEstimator"]


class MCRANoiseEstimator:
    """最小值控制的递归平均噪声估计器。

    算法流程（每帧）：

    1. 频率轴平滑 → 时间轴递归平滑，得到平滑功率谱 ``S``；
    2. 在长度 ``L`` 帧的滑动窗内跟踪 ``S`` 的最小值 ``S_min``；
    3. 若 ``S / S_min > delta``，判该频点存在语音；
    4. 语音存在概率 ``p`` 递归平滑后，用它调制噪声更新速率：
       有语音就几乎不更新，没语音就快速跟踪。

    Args:
        alpha_s: 功率谱时间平滑系数。越大越平滑、跟踪越慢。
        alpha_p: 语音存在概率的平滑系数。
        alpha_d: 噪声更新的基础平滑系数。
        delta: 判定语音存在的 ``S/S_min`` 门限。经典取值 5（约 7 dB）。
        L: 最小值跟踪窗长（帧）。默认 125 帧 ≈ 2 秒 @16 ms 帧移 ——
            必须明显长于最长的连续语音段，否则最小值会被语音抬高，
            导致噪声被高估、语音被过度削弱。
    """

    def __init__(
        self,
        cfg: STFTConfig = DEFAULT_CONFIG,
        alpha_s: float = 0.8,
        alpha_p: float = 0.2,
        alpha_d: float = 0.95,
        delta: float = 5.0,
        L: int = 125,
    ) -> None:
        self.cfg = cfg
        self.alpha_s = alpha_s
        self.alpha_p = alpha_p
        self.alpha_d = alpha_d
        self.delta = delta
        self.L = L
        # 频率轴平滑窗（归一化 Hann，宽度 5）。作用是抑制单个频点的随机起伏，
        # 否则最小值跟踪会被谱谷噪声带偏。
        w = np.hanning(5)
        self._bfreq = w / w.sum()
        self.reset()

    def reset(self) -> None:
        n = self.cfg.n_freq
        self._S = np.zeros(n)
        self._S_min = np.zeros(n)
        self._S_tmp = np.zeros(n)
        self._p = np.zeros(n)
        self._noise = np.zeros(n)
        self._frame_idx = 0
        self._initialized = False

    @property
    def noise_power(self) -> np.ndarray:
        """当前噪声功率谱估计 ``(n_freq,)``。"""
        return self._noise

    @property
    def speech_prob(self) -> np.ndarray:
        """每个频点的语音存在概率 ``(n_freq,)``（可视化很直观）。"""
        return self._p

    def update(self, power: np.ndarray) -> np.ndarray:
        """喂入一帧功率谱 ``|Y|^2``，返回更新后的噪声功率谱估计。"""
        power = np.asarray(power, dtype=np.float64).reshape(-1)

        if not self._initialized:
            # 首帧直接用观测值初始化。不假设它是纯噪声 ——
            # 就算首帧全是语音，MCRA 也会在约 L 帧内把估计拉回正确水平。
            self._S = power.copy()
            self._S_min = power.copy()
            self._S_tmp = power.copy()
            self._noise = power.copy()
            self._initialized = True
            self._frame_idx = 1
            return self._noise

        # 1) 频率轴平滑 + 时间轴递归平滑
        sf = np.convolve(power, self._bfreq, mode="same")
        self._S = self.alpha_s * self._S + (1.0 - self.alpha_s) * sf

        # 2) 滑动窗最小值跟踪。
        #    用双缓冲实现 O(1) 更新：S_tmp 累积当前窗内的最小值，
        #    每 L 帧把它交换进 S_min 并重置。这样 S_min 反映的始终是
        #    最近 L~2L 帧的最小值，无需保存整个窗的历史。
        if self._frame_idx % self.L == 0:
            self._S_min = np.minimum(self._S_tmp, self._S)
            self._S_tmp = self._S.copy()
        else:
            self._S_min = np.minimum(self._S_min, self._S)
            self._S_tmp = np.minimum(self._S_tmp, self._S)

        # 3) 语音存在判决：平滑谱显著高于局部最小值 → 有语音
        ratio = self._S / np.maximum(self._S_min, 1e-12)
        indicator = (ratio > self.delta).astype(np.float64)

        # 4) 概率平滑后调制噪声更新速率
        self._p = self.alpha_p * self._p + (1.0 - self.alpha_p) * indicator
        alpha_tilde = self.alpha_d + (1.0 - self.alpha_d) * self._p
        self._noise = alpha_tilde * self._noise + (1.0 - alpha_tilde) * power

        self._frame_idx += 1
        return self._noise
