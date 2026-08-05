"""STFT / iSTFT —— 整条链路的地基。

三条硬约束，任何一条被破坏，后面所有指标都失去意义：

1. **完美重构**：不做任何处理时 ``istft(stft(x)) == x``。
   谱减、维纳、CRM 掩码本质上都是"改幅度/复数谱再重建"。如果重建本身带误差，
   算出来的 SI-SDR 提升里就混进了系统误差，分不清是算法的功劳还是框架的 bug。

2. **因果**：不使用 center padding。``librosa.stft`` 默认 ``center=True``，
   会在信号两端各补半个窗，等价于让第 0 帧偷看了未来 16 ms。
   离线评测时这只是个小偏差，但流式部署时根本做不到，会导致
   "离线指标很好、上线就崩"的经典事故。这里一律左侧补零、右对齐。

3. **离线 / 流式逐样本一致**：``StreamingSTFT`` 逐块喂入的结果必须与 ``stft()``
   整段计算的结果完全相同。这条由 ``tests/test_stft.py`` 强制，
   它是后面 ONNX 流式一致性校验能够成立的前提。

窗函数选择
----------
分析窗与合成窗同为 **sqrt(周期 Hann)**。周期 Hann 在 hop = N/2 时满足
``sum_k hann[n - k*H] == 1``（COLA），因此分析窗 × 合成窗 = Hann，
重叠相加增益恒为 1，**不需要做任何归一化除法**。

这一点对流式很关键：流式 iSTFT 看不到全局窗和，做不了归一化除法。
只有当稳态增益天然为 1 时，流式与离线才可能逐样本相同。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rtse import HOP_LENGTH, N_FFT

__all__ = [
    "STFTConfig",
    "StreamingSTFT",
    "StreamingISTFT",
    "stft",
    "istft",
    "sqrt_hann",
    "check_cola",
    "num_frames",
]


def _cola_profile(win: np.ndarray, hop: int) -> np.ndarray:
    """稳态区内 ``sum_k win[n - k*hop]`` 的取值（长度 hop）。

    重叠相加的稳态增益是以 hop 为周期的，所以只需看一个 hop 的长度。
    """
    n_overlap = win.size // hop
    acc = np.zeros(hop, dtype=np.float64)
    for k in range(n_overlap):
        acc += win[k * hop : (k + 1) * hop]
    return acc


def sqrt_hann(n_fft: int, hop: int | None = None) -> np.ndarray:
    """周期 Hann 窗的平方根，可选做 COLA 归一化。

    注意是 **周期** 版 ``0.5 - 0.5*cos(2*pi*n/N)``（n = 0..N-1），
    不是对称版 ``np.hanning`` 的 ``2*pi*n/(N-1)``。对称版不满足 COLA，
    用它会引入约 1e-3 量级的重构误差 —— 刚好小到不容易发现、
    又大到足以污染 SI-SDR。

    关于归一化：sqrt-Hann 的"分析窗 × 合成窗 = Hann"，而 Hann 的重叠相加增益
    等于 ``n_fft / (2*hop)`` —— 只有在 50% 重叠（hop = N/2）时它才恰好是 1。
    75% 重叠时是 2，直接用会让重建信号整体放大一倍。
    传入 ``hop`` 后，窗会被除以 ``sqrt(增益)``，使任意合法 hop 下的 OLA 增益都为 1。
    （hop = N/2 时该系数正好是 1，因此默认配置的数值不受任何影响。）
    """
    n = np.arange(n_fft, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)
    win = np.sqrt(hann)
    if hop is None:
        return win

    gain = _cola_profile(hann, hop)
    # Hann 的 COLA 增益是常数，若不是常数说明该 hop 根本不满足 COLA
    if np.max(np.abs(gain - gain[0])) > 1e-12:
        raise ValueError(f"Hann 窗在 hop={hop} 下不满足 COLA（增益非常数），无法做完美重构")
    return win / np.sqrt(gain[0])


def check_cola(n_fft: int = N_FFT, hop: int = HOP_LENGTH) -> float:
    """返回归一化 sqrt-Hann 的分析×合成窗在给定 hop 下的 COLA 偏差。

    用于自检：返回值应当 < 1e-12。``rtse-doctor`` 会调用它。
    """
    win = sqrt_hann(n_fft, hop)
    prod = win * win  # 分析窗 × 合成窗
    return float(np.max(np.abs(_cola_profile(prod, hop) - 1.0)))


@dataclass(frozen=True)
class STFTConfig:
    """STFT 参数。整个项目共享同一份配置实例，避免各模块参数漂移。"""

    n_fft: int = N_FFT
    hop: int = HOP_LENGTH

    def __post_init__(self) -> None:
        if self.n_fft % self.hop != 0:
            raise ValueError(
                f"n_fft({self.n_fft}) 必须是 hop({self.hop}) 的整数倍，否则 COLA 不成立"
            )
        dev = check_cola(self.n_fft, self.hop)
        if dev > 1e-12:
            raise ValueError(f"sqrt-Hann 在 hop={self.hop} 下不满足 COLA，偏差 {dev:.3e}")

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def pad(self) -> int:
        """左侧补零量。

        取 ``n_fft - hop``，使原始第 0 个样本也能被完整的 ``n_fft/hop`` 个窗覆盖，
        从而落在 COLA 稳态区内。这也正好等于流式 iSTFT 的 overlap 缓冲长度，
        两者必须相同，离线/流式才能对齐。
        """
        return self.n_fft - self.hop

    @property
    def window(self) -> np.ndarray:
        return sqrt_hann(self.n_fft, self.hop)

    @property
    def latency_samples(self) -> int:
        """分析-修改-合成链路的算法延迟（样本数）。

        某个样本进入缓冲后，要等它所属的最后一个分析窗凑齐才能被完整重建，
        即最坏情况需要等待一整个窗长。这是**理论值**；
        ``rtse.runtime.latency`` 里有对实际实现的经验测量，两者应当吻合。
        """
        return self.n_fft


DEFAULT_CONFIG = STFTConfig()


def num_frames(n_samples: int, cfg: STFTConfig = DEFAULT_CONFIG) -> int:
    """给定样本数所需的帧数。

    需要覆盖到填充后坐标 ``pad + n_samples - 1``，且该位置必须被 ``floor(p/hop)``
    这一帧包含，故帧数 = floor((pad + n - 1) / hop) + 1。
    """
    if n_samples <= 0:
        return 0
    last = cfg.pad + n_samples - 1
    return last // cfg.hop + 1


def stft(x: np.ndarray, cfg: STFTConfig = DEFAULT_CONFIG) -> np.ndarray:
    """整段 STFT。

    Args:
        x: 一维实信号 ``(n_samples,)``。
    Returns:
        复数谱 ``(n_frames, n_freq)``，dtype complex128。
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    n_fr = num_frames(n, cfg)
    if n_fr == 0:
        return np.zeros((0, cfg.n_freq), dtype=np.complex128)

    # 右侧补到能容纳最后一帧
    total = (n_fr - 1) * cfg.hop + cfg.n_fft
    y = np.zeros(total, dtype=np.float64)
    y[cfg.pad : cfg.pad + n] = x

    # 用 stride trick 一次性取出所有帧，避免 Python 层循环
    frames = np.lib.stride_tricks.sliding_window_view(y, cfg.n_fft)[:: cfg.hop]
    return np.fft.rfft(frames * cfg.window, axis=-1)


def istft(
    spec: np.ndarray, length: int | None = None, cfg: STFTConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """整段 iSTFT（纯重叠相加，**不做窗和归一化**）。

    不归一化是刻意的：稳态 COLA 增益恒为 1，而流式版本没有能力做全局归一化。
    只有两边都用裸 OLA，离线与流式才可能逐样本一致。
    有效区间之外（左侧 pad 部分）的边缘样本会被裁掉，不影响结果。

    Args:
        spec: ``(n_frames, n_freq)`` 复数谱。
        length: 原始样本数。为 None 时按帧数反推（可能多出不足一帧的尾巴）。
    """
    spec = np.asarray(spec)
    n_fr = spec.shape[0]
    if n_fr == 0:
        return np.zeros(0, dtype=np.float64)

    frames = np.fft.irfft(spec, n=cfg.n_fft, axis=-1) * cfg.window

    total = (n_fr - 1) * cfg.hop + cfg.n_fft
    y = np.zeros(total, dtype=np.float64)
    for i in range(n_fr):
        s = i * cfg.hop
        y[s : s + cfg.n_fft] += frames[i]

    out = y[cfg.pad :]
    if length is not None:
        if length <= out.size:
            out = out[:length]
        else:
            out = np.pad(out, (0, length - out.size))
    return out


def spec_ref(cfg: STFTConfig = DEFAULT_CONFIG) -> float:
    """满幅正弦在其所属频点上的 STFT 幅度，用作 dB 的 0 点参考。

    **为什么必须显式定义这个参考**：``|rfft(frame * window)|`` 是 512 个加窗样本的
    相干求和，量级远大于时域幅度 —— 一个峰值 1.0 的正弦在对应 bin 上的幅度约是 160，
    即 +44 dB，而不是 0 dB。若按"时域信号在 ±1 之间所以 dB 应该是负的"这个直觉
    去定色标范围（比如 -85~-5 dB），整幅频谱图会全部压在色标顶端，
    看上去是一片均匀的亮黄色，什么结构都看不出来（见 docs/ISSUES.md I-12）。

    归一化后：满幅正弦 = 0 dBFS，正常说话的语音峰值大致落在 -25 ~ -45 dBFS，
    安静环境底噪在 -80 dBFS 附近 —— 色标范围就有了物理含义。
    """
    return float(np.sum(cfg.window)) / 2.0


def magnitude_db(
    spec: np.ndarray, cfg: STFTConfig = DEFAULT_CONFIG, floor_db: float = -110.0
) -> np.ndarray:
    """复数谱 → dBFS 幅度谱。

    实时管线、文件实验台、评测器**必须都调用这一个函数**。
    各处自己写 ``20*log10(abs(spec))`` 是频谱图对不上的头号原因。
    """
    return np.maximum(
        20.0 * np.log10(np.abs(spec) / spec_ref(cfg) + 1e-12), floor_db
    )


class StreamingSTFT:
    """流式分析：每次喂入 ``hop`` 个样本，吐出一帧复数谱。

    内部缓冲初始化为全零、长度 ``n_fft``，这与离线版左侧补 ``n_fft - hop`` 个零
    在数值上完全等价 —— 第一次 push 后缓冲内容恰好是
    ``[zeros(n_fft - hop), x[0:hop]]``，与离线第 0 帧逐样本相同。
    """

    def __init__(self, cfg: STFTConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg
        self._win = cfg.window
        self._buf = np.zeros(cfg.n_fft, dtype=np.float64)

    def reset(self) -> None:
        self._buf[:] = 0.0

    def push(self, block: np.ndarray) -> np.ndarray:
        """喂入恰好 ``hop`` 个样本，返回 ``(n_freq,)`` 复数谱。"""
        block = np.asarray(block, dtype=np.float64).reshape(-1)
        if block.size != self.cfg.hop:
            raise ValueError(f"每次必须喂入恰好 hop={self.cfg.hop} 个样本，收到 {block.size}")
        # 左移一个 hop，尾部放入新数据
        self._buf[: -self.cfg.hop] = self._buf[self.cfg.hop :]
        self._buf[-self.cfg.hop :] = block
        return np.fft.rfft(self._buf * self._win)


class StreamingISTFT:
    """流式合成：每次喂入一帧复数谱，吐出 ``hop`` 个时域样本。

    输出的是**填充坐标系**下的样本流，即比原始信号超前 ``pad`` 个样本。
    换句话说，前 ``pad`` 个输出样本对应离线版被裁掉的左侧补零区，
    调用方（``rtse.runtime``）负责丢弃它们。这 ``pad`` 个样本正是算法延迟的来源。
    """

    def __init__(self, cfg: STFTConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg
        self._win = cfg.window
        self._tail = np.zeros(cfg.n_fft - cfg.hop, dtype=np.float64)

    def reset(self) -> None:
        self._tail[:] = 0.0

    def push(self, spec: np.ndarray) -> np.ndarray:
        """喂入 ``(n_freq,)`` 复数谱，返回 ``(hop,)`` 时域样本。"""
        spec = np.asarray(spec).reshape(-1)
        if spec.size != self.cfg.n_freq:
            raise ValueError(f"谱长度应为 {self.cfg.n_freq}，收到 {spec.size}")
        frame = np.fft.irfft(spec, n=self.cfg.n_fft) * self._win
        frame[: self._tail.size] += self._tail
        out = frame[: self.cfg.hop].copy()
        self._tail = frame[self.cfg.hop :]
        return out
