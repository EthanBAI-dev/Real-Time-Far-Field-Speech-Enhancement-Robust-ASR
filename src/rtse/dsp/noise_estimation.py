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


class _MinTracker:
    """滑动窗最小值跟踪，双缓冲实现 O(1) 更新。

    每 ``L`` 帧把累积的候选最小值交换进当前估计并重置。这样任意时刻的估计
    反映的都是最近 ``L`` ~ ``2L`` 帧内的最小值，不需要保存整个窗的历史。

    单独抽成一个类是因为 :class:`MCRANoiseEstimator` 需要**两个不同时间尺度**
    的实例（见 I-20 的修复说明），复制粘贴同一段双缓冲逻辑两遍很容易在后续
    修改时只改一处、让两份实现悄悄跑偏。
    """

    def __init__(self, n_freq: int, L: int) -> None:
        self.L = L
        self._min = np.zeros(n_freq)
        self._tmp = np.zeros(n_freq)
        self._idx = 0

    def reset(self, init: np.ndarray) -> None:
        self._min = init.copy()
        self._tmp = init.copy()
        self._idx = 0

    def update(self, s: np.ndarray) -> np.ndarray:
        if self._idx > 0 and self._idx % self.L == 0:
            self._min = np.minimum(self._tmp, s)
            self._tmp = s.copy()
        else:
            self._min = np.minimum(self._min, s)
            self._tmp = np.minimum(self._tmp, s)
        self._idx += 1
        return self._min


class MCRANoiseEstimator:
    """最小值控制的递归平均噪声估计器。

    算法流程（每帧）：

    1. 频率轴平滑 → 时间轴递归平滑，得到平滑功率谱 ``S``；
    2. 用两个不同时间尺度的滑动窗跟踪 ``S`` 的最小值，取二者中更低的作为地板；
    3. 若 ``S / floor > delta``，判该频点存在语音；
    4. 语音存在概率 ``p`` 递归平滑后，用它调制噪声更新速率：
       有语音就几乎不更新，没语音就快速跟踪。

    Args:
        alpha_s: 功率谱时间平滑系数。越大越平滑、跟踪越慢。
        alpha_p: 语音存在概率的平滑系数。
        alpha_d: 噪声更新的基础平滑系数。
        delta: 判定语音存在的 ``S/floor`` 门限。经典取值 5（约 7 dB）。
        L: 短时间尺度最小值跟踪窗长（帧）。默认 125 帧 ≈ 2 秒 @16 ms 帧移，
            负责快速适应变化的噪声（比如噪声突然变大）。
        L2_mult: 长时间尺度窗长相对 ``L`` 的倍数，默认 8（≈16 秒）。
            负责兜底找到真正的噪声地板——见下方"双时间尺度"说明。

    双时间尺度的由来（见 ``docs/ISSUES.md`` I-20）
    -----------------------------------------------
    单一 ``L=125`` 帧（2 秒）窗口在**混响 + 真实连续语音**（词间停顿短、
    又被混响尾巴填上残余能量）的场景下会系统性失败：2 秒内可能根本没有出现过
    足够安静的时刻，``S_min`` 被"卡"在一个远高于真实噪声地板的值上，
    且直到下一次 2 秒重置前都无法修正。SNR 越高（真实噪声地板越低），
    这个偏差在**相对**幅度上越夸张——实测 SNR=20 dB 时噪声功率被高估 22 dB
    （约 170 倍），导致维纳滤波之类的增益函数把大半语音削掉。

    延长单一窗口的做法（比如直接把 ``L`` 调到 2000 帧）会牺牲对**真实**噪声突变
    的响应速度。这里保留原有的短窗口不变（负责噪声突变的正常适应），
    额外并行跑一个长窗口只用来兜底"短窗口一直没找到过低点"这种情况，
    取两者较小值作为最终地板。代价是长窗口重置前的 ~16 秒内，如果噪声地板
    真的持续上升，判决会偏保守（更容易把新的噪声误判成语音）——这是刻意的
    权衡：宁可短暂偏保守，也不要回到"直接按几个数量级高估噪声"的状态。
    """

    def __init__(
        self,
        cfg: STFTConfig = DEFAULT_CONFIG,
        alpha_s: float = 0.8,
        alpha_p: float = 0.2,
        alpha_d: float = 0.95,
        delta: float = 5.0,
        L: int = 125,
        L2_mult: int = 8,
    ) -> None:
        self.cfg = cfg
        self.alpha_s = alpha_s
        self.alpha_p = alpha_p
        self.alpha_d = alpha_d
        self.delta = delta
        self.L = L
        self.L2_mult = L2_mult
        # 频率轴平滑窗（归一化 Hann，宽度 5）。作用是抑制单个频点的随机起伏，
        # 否则最小值跟踪会被谱谷噪声带偏。
        w = np.hanning(5)
        self._bfreq = w / w.sum()
        self._short = _MinTracker(cfg.n_freq, L)
        self._long = _MinTracker(cfg.n_freq, L * L2_mult)
        self.reset()

    def reset(self) -> None:
        n = self.cfg.n_freq
        self._S = np.zeros(n)
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
            self._short.reset(power)
            self._long.reset(power)
            self._noise = power.copy()
            self._initialized = True
            self._frame_idx = 1
            return self._noise

        # 1) 频率轴平滑 + 时间轴递归平滑
        sf = np.convolve(power, self._bfreq, mode="same")
        self._S = self.alpha_s * self._S + (1.0 - self.alpha_s) * sf

        # 2) 双时间尺度最小值跟踪，取更低者
        s_min_short = self._short.update(self._S)
        s_min_long = self._long.update(self._S)
        floor = np.minimum(s_min_short, s_min_long)

        # 3) 语音存在判决：平滑谱显著高于地板 → 有语音
        ratio = self._S / np.maximum(floor, 1e-12)
        indicator = (ratio > self.delta).astype(np.float64)

        # 4) 概率平滑后调制噪声更新速率
        self._p = self.alpha_p * self._p + (1.0 - self.alpha_p) * indicator
        alpha_tilde = self.alpha_d + (1.0 - self.alpha_d) * self._p
        self._noise = alpha_tilde * self._noise + (1.0 - alpha_tilde) * power

        self._frame_idx += 1
        return self._noise
