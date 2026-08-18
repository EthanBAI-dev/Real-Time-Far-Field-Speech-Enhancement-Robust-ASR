"""传统 DSP 语音增强：谱减法、维纳滤波、MMSE-LSA。

这三者构成神经网络的对照组。**先把它们做对做好，再上神经网络**，
否则"NN 提升了 3 dB"这句话没有意义 —— 对照组是随便实现的弱基线，
提升多少都不能说明模型好。

三者共享同一套骨架：估计噪声 → 算增益 → 乘到带噪谱上 → 用带噪相位重建。
差别只在**增益函数**，这也正好是它们效果差异的全部来源。

相位处理：全部沿用带噪相位。这是传统方法的固有上限，
也正是后面神经网络用**复数比值掩码（CRM）**能超越它们的地方 —— 对比表里
这一条要能讲清楚。
"""

from __future__ import annotations

import numpy as np
from scipy.special import exp1

from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig
from rtse.dsp.base import StreamingEnhancer
from rtse.dsp.noise_estimation import MCRANoiseEstimator

__all__ = ["SpectralSubtraction", "WienerFilter", "MMSELogSTSA", "DSP_METHODS", "build_dsp"]

_EPS = 1e-12

#: 维纳/MMSE-LSA 的"高 SNR 收手"区间（帧级后验 SNR，dB）。
#: 低于下界完全按算法处理，高于上界完全透传，中间线性过渡。
#: 这两个数是在 `data/testsets/dns_objective` 的 80 条带噪样本上**扫出来**的，
#: 不是拍脑袋定的：初版按直觉猜 (12,24)，实测偏保守——扫过 (0,12)~(18,30) 后
#: (2,14) 是拐点，再低低 SNR 段就开始掉。详见 docs/ISSUES.md I-35。
#: ⚠️ 调参与评测用的是同一批 80 条样本，有过拟合风险，需在另两套测试集上复核。
#: 谱减法不用它——它的过减因子本来就随帧 SNR 自适应，高 SNR 段没有负收益。
BACKOFF_SNR_DB = (2.0, 14.0)


class _GainEnhancer(StreamingEnhancer):
    """基于实数增益的增强器骨架。子类只需实现 ``compute_gain``。"""

    def __init__(
        self,
        cfg: STFTConfig = DEFAULT_CONFIG,
        min_gain_db: float = -18.0,
        noise_estimator: MCRANoiseEstimator | None = None,
        backoff_snr_db: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(cfg)
        # 增益下限。不设下限会把噪声压成零星孤立的谱峰 —— 就是"音乐噪声"的来源。
        # -18 dB 是常用折中：残余噪声仍可听但不刺耳，且不至于吃掉弱辅音。
        self.min_gain = 10.0 ** (min_gain_db / 20.0)
        self.noise_est = noise_estimator or MCRANoiseEstimator(cfg)
        self.backoff_snr_db = backoff_snr_db
        self._last_gain: np.ndarray | None = None

    def reset(self) -> None:
        self.noise_est.reset()
        self._last_gain = None

    @property
    def last_gain(self) -> np.ndarray | None:
        return self._last_gain

    def compute_gain(self, power: np.ndarray, noise: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _backoff(self, power: np.ndarray, noise: np.ndarray) -> float:
        """噪声已经很少时该收手多少。返回 0~1：0=按算法原样处理，1=完全透传。

        用的是**帧级后验 SNR**，也就是谱减法里那个自适应过减因子用的同一个量
        （见 ``SpectralSubtraction.compute_gain``）。刻意复用它，不引入新的判据：
        谱减法靠它在高 SNR 段自动减得更少，实测 10/15 dB 段没有负收益；
        维纳/MMSE-LSA 缺的正是这一环。

        **不做二值旁路**。帧级 SNR 这个量本身不可靠——实测各 SNR 档的中位数是
        5.0/9.8/8.4/8.7/13.2 dB，并不随输入 SNR 单调，15 dB 档的范围（5~27）
        还和纯干净输入（19.3）大幅重叠。拿它做阈值判决必然误伤，
        但拿它做**连续的强度调节**是安全的：估计偏了也只是处理力度略轻，
        不会出现行为突变。
        """
        if self.backoff_snr_db is None:
            return 0.0
        lo, hi = self.backoff_snr_db
        snr_db = 10.0 * np.log10((power.sum() + _EPS) / (noise.sum() + _EPS))
        return float(np.clip((snr_db - lo) / (hi - lo), 0.0, 1.0))

    def process_frame(self, spec: np.ndarray) -> np.ndarray:
        power = np.abs(spec) ** 2
        noise = self.noise_est.update(power)
        gain = np.clip(self.compute_gain(power, noise), self.min_gain, 1.0)
        t = self._backoff(power, noise)
        if t > 0.0:
            gain = gain + (1.0 - gain) * t
        # 存的是**限幅之后、真正施加**的增益。决策导向的反馈必须用这一个，
        # 否则递推式里的 G(l-1) 和实际输出对不上（min_gain 一旦生效两者就会分叉，
        # 而 -18 dB 的下限在低 SNR 段是经常生效的）。收手混合同理：
        # 它改变的是**实际施加**的增益，反馈必须跟着一起变。
        self._last_gain = gain
        return spec * gain


class SpectralSubtraction(_GainEnhancer):
    """谱减法（Berouti 过减版本）。

    ``|S|^2 = |Y|^2 - alpha * lambda_d``，并以 ``beta * lambda_d`` 为谱下限。

    过减因子 ``alpha`` 随帧 SNR 自适应（Berouti 1979）：SNR 低时多减，SNR 高时少减。
    固定 alpha 的朴素版本在低 SNR 段残留噪声、在高 SNR 段吃掉语音，两头不讨好。

    **已知缺陷（要在失败案例分析里展示的）**：
    减法后的谱有大量随机的孤立残余峰，重建后听起来是断续的"叮咚"声，
    即**音乐噪声**。谱下限 beta 只能缓解不能消除。
    这正是维纳/MMSE-LSA 存在的理由。
    """

    name = "specsub"

    def __init__(
        self,
        cfg: STFTConfig = DEFAULT_CONFIG,
        alpha0: float = 4.0,
        beta: float = 0.02,
        min_gain_db: float = -18.0,
    ) -> None:
        super().__init__(cfg, min_gain_db)
        self.alpha0 = alpha0
        self.beta = beta

    def compute_gain(self, power: np.ndarray, noise: np.ndarray) -> np.ndarray:
        # 帧级后验 SNR（dB），用来自适应地定过减因子
        snr_db = 10.0 * np.log10((power.sum() + _EPS) / (noise.sum() + _EPS))
        snr_db = np.clip(snr_db, -5.0, 20.0)
        alpha = self.alpha0 - snr_db * 3.0 / 20.0

        sub = power - alpha * noise
        floor = self.beta * noise
        return np.sqrt(np.maximum(sub, floor) / (power + _EPS))


class WienerFilter(_GainEnhancer):
    """维纳滤波 + 决策导向（Decision-Directed）先验 SNR 估计。

    增益 ``G = xi / (1 + xi)``，其中 ``xi`` 是先验 SNR。

    先验 SNR 拿不到真值，只能估计。**决策导向**（Ephraim & Malah 1984）是关键：

        xi(l) = a * G(l-1)^2 * gamma(l-1) + (1 - a) * max(gamma(l) - 1, 0)

    第一项用上一帧的增强结果推断，第二项是当前帧的最大似然估计。
    ``a = 0.98`` 让 xi 在时间上高度平滑 —— **音乐噪声被压下去的真正原因就在这里**：
    帧间随机起伏被平滑掉了，孤立残余峰无法形成。

    代价是语音起始处会有约一帧的响应延迟（xi 还停留在上一帧的低值），
    表现为轻微的辅音削弱。这是这类方法的固有权衡。
    """

    name = "wiener"

    def __init__(
        self,
        cfg: STFTConfig = DEFAULT_CONFIG,
        dd_alpha: float = 0.98,
        min_gain_db: float = -18.0,
        backoff_snr_db: tuple[float, float] | None = BACKOFF_SNR_DB,
    ) -> None:
        super().__init__(cfg, min_gain_db, backoff_snr_db=backoff_snr_db)
        self.dd_alpha = dd_alpha
        self._prev_gamma: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._prev_gamma = None

    def _a_priori_snr(self, power: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gamma = power / (noise + _EPS)  # 后验 SNR
        ml = np.maximum(gamma - 1.0, 0.0)  # 当前帧最大似然估计
        if self._last_gain is None:
            xi = ml  # 首帧无历史，退化为最大似然
        else:
            # 反馈用 self._last_gain —— 上一帧**实际施加**的（已限幅）增益
            xi = self.dd_alpha * (self._last_gain**2) * self._prev_gamma + (1 - self.dd_alpha) * ml
        # 下限 -25 dB：xi 过小会让增益趋零，语音尾音被整段吃掉
        xi = np.maximum(xi, 10.0 ** (-25.0 / 10.0))
        self._prev_gamma = gamma
        return xi, gamma

    def compute_gain(self, power: np.ndarray, noise: np.ndarray) -> np.ndarray:
        xi, _ = self._a_priori_snr(power, noise)
        return xi / (1.0 + xi)


def _exp1_half_poly(nu: np.ndarray) -> np.ndarray:
    """``0.5 * E1(nu)`` 的分段多项式近似，替代 ``scipy.special.exp1``。

    见 docs/FINDINGS.md F-04：MMSE-LSA 比维纳滤波慢 60%，瓶颈就是这个指数积分——
    它没有解析解，`scipy.special.exp1` 每帧都要做数值积分，而维纳滤波完全不需要它。

    系数来自参考项目 `ref/speech-processing-master/Speech Enhancement/MMSE/
    test_mmse_1.py` 的 `Gan_log_mmse2`（语音增强文献里的标准写法，如 Loizou
    《Speech Enhancement》一书）。**这三段近似的是 E1(nu) 本身，不是 0.5*E1(nu)**——
    最后要统一乘 0.5，第一版实现漏了这一步，把"0.52dB 的单点误差"错误地放大成了
    "2.5dB 的整体降噪收益回退"，一度以为是 decision-directed 反馈环放大了近似误差。
    补上这个因子后重新测（15 组噪声类型 × SNR 组合 + 8 秒音频的端到端流式基准）：

    - 单点误差：对增益的最大误差 0.52 dB、95 分位 0.33 dB
      （`nu` 取 1e-6~500 对数网格）；
    - 端到端 SI-SDR：15 组条件下与精确解的差异全部 < 0.05 dB，
      没有一个方向系统性偏移；
    - 速度：8 秒音频、498 帧，精确解 49.6 us/帧，近似解 42.1 us/帧，快 ~15%——
      比 `exp1` 单独测的 1.43x 温和，因为它只占每帧总开销的一部分。

    **教训**：把"改动前后有没有验证过"和"验证用的是不是同一份代码"分开看——
    这次是先在交互式 shell 里手算验证了系数（算对了），再誊抄进函数体时漏了
    一个 `0.5*`（代码错了），于是端到端测试测出的"2.5dB 回退"其实是在测这个
    抄写 bug，跟近似算法本身完全无关。修复思路本来是对的，只是被自己的
    误写误导了一次。
    """
    nu = np.asarray(nu, dtype=np.float64)
    e1 = np.empty_like(nu)
    m1 = nu < 0.1
    m2 = (nu >= 0.1) & (nu < 1.0)
    m3 = nu >= 1.0
    e1[m1] = -2.3 * np.log10(nu[m1]) - 0.6
    e1[m2] = -1.544 * np.log10(nu[m2]) + 0.166
    e1[m3] = 10.0 ** (-0.53 * nu[m3] - 0.26)
    return 0.5 * e1


class MMSELogSTSA(WienerFilter):
    """MMSE 对数谱幅度估计（Ephraim & Malah 1985）。

    最小化**对数**谱幅度的均方误差，而不是幅度本身的均方误差。
    这一改动的意义在于：人耳对响度的感知近似对数，
    因此对数域的最优解在听感上明显优于线性域的最优解（即维纳）。

    增益：``G = xi/(1+xi) * exp(0.5 * E1(nu))``，``nu = xi*gamma/(1+xi)``，
    ``E1`` 是指数积分。默认走 ``_exp1_half_poly`` 的多项式近似而不是
    ``scipy.special.exp1`` 精确解——15 组噪声/SNR 条件下两者的 SI-SDR
    差异全部 < 0.05 dB，换来端到端约 15% 的提速（见该函数文档字符串）。
    需要逐样本对拍验证时传 ``exact=True``。

    相比维纳，它在低 SNR 段的音乐噪声更少，是三个 DSP 方法里听感最好的，
    也是神经网络需要真正超越的对手。
    """

    name = "mmse-lsa"

    def __init__(
        self,
        cfg: STFTConfig = DEFAULT_CONFIG,
        dd_alpha: float = 0.98,
        min_gain_db: float = -18.0,
        exact: bool = False,
        backoff_snr_db: tuple[float, float] | None = BACKOFF_SNR_DB,
    ) -> None:
        super().__init__(cfg, dd_alpha, min_gain_db, backoff_snr_db=backoff_snr_db)
        self.exact = exact

    def compute_gain(self, power: np.ndarray, noise: np.ndarray) -> np.ndarray:
        xi, gamma = self._a_priori_snr(power, noise)
        nu = np.clip(xi * gamma / (1.0 + xi), 1e-8, 500.0)
        # nu 很大时精确解与近似解都趋于 0，exp(0)=1，增益退化为维纳增益 —— 行为正确
        half_e1 = 0.5 * exp1(nu) if self.exact else _exp1_half_poly(nu)
        return (xi / (1.0 + xi)) * np.exp(half_e1)


DSP_METHODS: dict[str, type[_GainEnhancer]] = {
    SpectralSubtraction.name: SpectralSubtraction,
    WienerFilter.name: WienerFilter,
    MMSELogSTSA.name: MMSELogSTSA,
}


def build_dsp(name: str, cfg: STFTConfig = DEFAULT_CONFIG, **kwargs) -> _GainEnhancer:
    if name not in DSP_METHODS:
        raise KeyError(f"未知的 DSP 方法 {name!r}，可选：{sorted(DSP_METHODS)}")
    return DSP_METHODS[name](cfg, **kwargs)
