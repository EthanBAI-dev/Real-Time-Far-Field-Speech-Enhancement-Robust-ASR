"""合成数据：噪声生成、房间冲激响应、按 SNR 混音。

**定位**：本地 bootstrap。在 Colab 侧的真实数据（DNS 噪声库、openSLR 真实 RIR）
到位之前，用它就能把整条链路跑通、把 Web 演示做出来、把测试写完。

真实数据到位后，这个模块**不会被丢弃**：
- ``mix_at_snr`` 是 Colab 侧合成脚本共用的（SNR 定义必须两边完全一致）；
- 合成噪声作为"受控噪声"档位保留，因为它的统计特性完全已知，
  便于做"白噪声下 SI-SDR 应该是多少"这类可解析验证。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from rtse import SAMPLE_RATE

__all__ = [
    "NOISE_KINDS",
    "make_noise",
    "make_rir",
    "mix_at_snr",
    "speech_active_mask",
    "apply_rir",
]

NOISE_KINDS = ("white", "pink", "brown", "babble", "car", "keyboard", "hum", "cafeteria")


def _pink(n: int, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    """1/f^(exponent/2) 幅度谱的有色噪声，用频域整形生成。"""
    n_fft = 1 << int(np.ceil(np.log2(max(n, 2))) + 1)
    spec = rng.standard_normal(n_fft // 2 + 1) + 1j * rng.standard_normal(n_fft // 2 + 1)
    freqs = np.arange(n_fft // 2 + 1, dtype=np.float64)
    freqs[0] = 1.0  # 避免直流除零
    spec /= freqs ** (exponent / 2.0)
    spec[0] = 0.0  # 去掉直流
    y = np.fft.irfft(spec, n=n_fft)[:n]
    return y / (np.std(y) + 1e-12)


def make_noise(kind: str, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """生成指定类型的噪声，输出单位方差。

    这几类覆盖了评测矩阵需要的谱形与时变特性：
    - ``white``：平坦谱，平稳。谱减法在它上面表现最好，是"最容易的一档"。
    - ``pink`` / ``brown``：低频占优，接近真实环境底噪。
    - ``babble`` / ``cafeteria``：多人说话叠加，**与语音谱高度重叠**，最难的一档，
      也是 STOI 和 ESTOI 会明显分道扬镳的地方。
    - ``car``：强低频 + 缓慢起伏。
    - ``keyboard``：冲激型、极度非平稳，专门用来打噪声估计器的跟踪速度。
    - ``hum``：50 Hz 工频及其谐波，窄带、极稳定。
    """
    rng = rng or np.random.default_rng()
    n = int(n_samples)

    if kind == "white":
        y = rng.standard_normal(n)
    elif kind == "pink":
        y = _pink(n, rng, exponent=1.0)
    elif kind == "brown":
        y = _pink(n, rng, exponent=2.0)
    elif kind in ("babble", "cafeteria"):
        # 多路"类语音"叠加：粉噪 × 音节速率（3~7 Hz）的幅度调制，
        # 再叠一个共振峰式的带通包络，逼近语音的长时谱。
        n_talker = 6 if kind == "babble" else 12
        y = np.zeros(n)
        t = np.arange(n) / SAMPLE_RATE
        for _ in range(n_talker):
            src = _pink(n, rng, exponent=1.2)
            syllable = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3.0, 7.0) * t + rng.uniform(0, 6.3))
            y += src * syllable
        if kind == "cafeteria":
            # 餐厅额外叠加稀疏的餐具碰撞冲激
            for _ in range(int(n / SAMPLE_RATE * 3)):
                p = rng.integers(0, max(1, n - 400))
                y[p : p + 400] += rng.standard_normal(400) * np.exp(-np.arange(400) / 40.0) * 3
    elif kind == "car":
        y = _pink(n, rng, exponent=2.5)
        t = np.arange(n) / SAMPLE_RATE
        y *= 1.0 + 0.3 * np.sin(2 * np.pi * 0.2 * t)  # 缓慢的发动机转速起伏
    elif kind == "keyboard":
        # 冲激序列：每秒约 6 次按键，每次是一个快速衰减的宽带脉冲
        y = np.zeros(n)
        for _ in range(max(1, int(n / SAMPLE_RATE * 6))):
            p = int(rng.integers(0, max(1, n - 800)))
            ln = 800
            y[p : p + ln] += rng.standard_normal(ln) * np.exp(-np.arange(ln) / 60.0)
    elif kind == "hum":
        t = np.arange(n) / SAMPLE_RATE
        y = sum(np.sin(2 * np.pi * 50 * h * t) / h for h in (1, 2, 3, 4, 5))
        y = y + 0.05 * rng.standard_normal(n)
    else:
        raise KeyError(f"未知噪声类型 {kind!r}，可选：{NOISE_KINDS}")

    return y / (np.std(y) + 1e-12)


def make_rir(
    t60: float,
    room_dim: tuple[float, float, float] = (6.0, 4.5, 2.8),
    src: tuple[float, float, float] = (1.5, 1.2, 1.6),
    mic: tuple[float, float, float] = (4.2, 3.1, 1.4),
    sr: int = SAMPLE_RATE,
    max_order: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """镜像源法（Image-Source Method）生成矩形房间的冲激响应。

    用镜像源法而不是"指数衰减白噪声"，是因为前者能产生**真实的早期反射结构**。
    远场语音的可懂度损失主要来自早期反射造成的谱着色和梳状滤波，
    而不只是混响尾巴的能量 —— 假 RIR 会让混响条件下的鲁棒性评测偏乐观。

    墙面反射系数由 Sabine 公式从目标 T60 反推：
    ``T60 = 0.161 * V / (S * a)`` → ``a``（吸声系数）→ ``beta = sqrt(1 - a)``。

    Args:
        t60: 目标混响时间（秒）。0 表示无混响（返回单位脉冲）。
        max_order: 镜像阶数。``None``（默认）时**按 t60 自动推算**：
            要让 RIR 覆盖到 t60 秒，声波最远要走 ``c * t60`` 米，
            对应阶数约 ``c * t60 / (2 * min(room_dim))``。

            **不要写死一个固定值。** 早期版本固定 ``max_order=12``，导致镜像源
            最远只到 0.56 秒，于是 **``t60`` 超过约 0.6 秒之后完全失效**——
            实测标称 0.6/0.9/1.0 秒的 RIR，实际 T60 全部落在 0.58~0.63 秒，
            即"更强的混响"根本没有被合成出来（见 docs/ISSUES.md I-22）。
    """
    rng = rng or np.random.default_rng()
    if t60 <= 0:
        return np.array([1.0])

    lx, ly, lz = room_dim
    volume = lx * ly * lz
    surface = 2 * (lx * ly + ly * lz + lz * lx)
    # Sabine 反推吸声系数，夹到物理合法区间
    absorption = float(np.clip(0.161 * volume / (surface * t60), 1e-3, 0.99))
    beta = np.sqrt(1.0 - absorption)

    c = 343.0
    n_rir = int(sr * (t60 * 1.2 + 0.05))
    s = np.array(src, dtype=np.float64)
    m = np.array(mic, dtype=np.float64)
    dims = np.array(room_dim, dtype=np.float64)

    if max_order is None:
        # 覆盖整个 n_rir 时间窗所需的阶数；+2 留一点余量。
        max_order = int(np.ceil(c * (n_rir / sr) / (2.0 * min(room_dim)))) + 2

    # **向量化 + 按 nx 分块**。不用三重 Python 循环：自适应阶数下 max_order
    # 可以到 60~100，而循环版是 O(max_order³ × 8) 次 Python 迭代 ——
    # 实测 order=12 就要 1 秒、order=60 要两分钟，数据合成阶段完全不可接受。
    #
    # 但也不能一次性把所有镜像源建成一个大数组：order=100 时
    # (201³ × 8 × 3) 个 float64 要 1.5 GB 以上，会直接把内存打满。
    # 折中是**外层对 nx 走 Python 循环（最多几百次，开销可忽略），
    # 内层对 ny/nz 向量化**，峰值内存降到 O(max_order²)，只有几 MB。
    axis = np.arange(-max_order, max_order + 1, dtype=np.float64)
    p_vec = np.array([[px, py, pz] for px in (0, 1) for py in (0, 1) for pz in (0, 1)],
                     dtype=np.float64)
    flip = 1 - 2 * p_vec  # (8, 3)
    rir = np.zeros(n_rir)

    ny_g, nz_g = np.meshgrid(axis, axis, indexing="ij")
    ny_f, nz_f = ny_g.ravel(), nz_g.ravel()

    for nx in axis:
        n_vec = np.stack([np.full(ny_f.size, nx), ny_f, nz_f], axis=1)  # (N, 3)
        # (N, 8, 3)：每个平移向量 × 8 种镜像翻转组合
        img = flip[None, :, :] * s[None, None, :] + 2.0 * n_vec[:, None, :] * dims
        dist = np.linalg.norm(img - m, axis=2).ravel()
        # 反射次数 = 各轴穿墙次数之和
        order_n = (np.abs(2.0 * n_vec[:, None, :] - p_vec[None, :, :]).sum(axis=2)
                   + p_vec.sum(axis=1)[None, :]).ravel()

        idx = np.rint(dist / c * sr).astype(np.int64)
        keep = (idx < n_rir) & (idx >= 0) & (dist > 1e-6)
        if not keep.any():
            continue
        amp = (beta ** order_n[keep]) / (4.0 * np.pi * dist[keep])
        np.add.at(rir, idx[keep], amp)  # 多个镜像源可能落在同一采样点，必须累加

    # 高频空气吸收：真实房间的混响尾巴高频衰减更快，不做这一步会显得"金属感"。
    # ⚠️ 这个包络自身等效 T60 约 3.45 秒，会和 Sabine 衰减叠加，
    # 使实际 T60 略低于目标值（长混响时更明显）—— 属于已知且可接受的偏差，
    # 用 rtse.dsp.rt60.estimate_t60() 可以量出实际值。
    rir *= np.exp(-np.arange(n_rir) / sr * 2.0)
    peak = np.max(np.abs(rir))
    return rir / peak if peak > 0 else rir


def apply_rir(x: np.ndarray, rir: np.ndarray, align_direct: bool = True) -> np.ndarray:
    """卷积房间冲激响应，输出长度与输入相同。

    **必须用 FFT 卷积，不能用 ``np.convolve``。** 这不是微优化，是量级差异：

    | T60 | RIR 长度 | ``np.convolve`` | ``fftconvolve`` | 倍数 |
    |---|---|---|---|---|
    | 0.3 s | 6560 | 42 ms | 1.7 ms | 25× |
    | 0.6 s | 12320 | **6996 ms** | 2.0 ms | **3482×** |
    | 0.9 s | 18080 | **7845 ms** | 2.1 ms | **3781×** |

    （4 秒音频段，实测）。直接卷积是 O(n·m)，长 RIR 下每个样本要几秒 ——
    训练时数据加载会彻底卡死，GPU 全程空转。这一条在 Colab 免费版
    （只有 2 个 vCPU）上尤其致命。见 docs/ISSUES.md I-19。

    ``align_direct=True`` 时按直达声（RIR 最大峰）对齐，消除传播时延。
    **这一步同样不能省**：不对齐的话混响信号相对干净参考整体滞后几十个样本，
    SI-SDR 会被这个纯时延压掉好几 dB，看起来像算法很差，实际只是没对齐。
    """
    y = fftconvolve(np.asarray(x, dtype=np.float64), np.asarray(rir, dtype=np.float64))
    if align_direct:
        y = y[int(np.argmax(np.abs(rir))) :]
    return y[: x.size] if y.size >= x.size else np.pad(y, (0, x.size - y.size))


def speech_active_mask(
    x: np.ndarray, frame: int = 512, hop: int = 256, rel_db: float = -35.0
) -> np.ndarray:
    """样本级的语音活跃掩码，用于"只在有语音处定义 SNR"。

    帧能量低于**全局峰值帧能量** rel_db 的帧判为静音。
    用相对门限而不是绝对门限，是为了不受录音音量的影响。
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n_fr = max(1, (x.size - frame) // hop + 1)
    energies = np.array([np.sum(x[i * hop : i * hop + frame] ** 2) for i in range(n_fr)])
    if energies.max() <= 0:
        return np.ones(x.size, dtype=bool)
    thresh = energies.max() * 10.0 ** (rel_db / 10.0)
    mask = np.zeros(x.size, dtype=bool)
    for i in np.flatnonzero(energies >= thresh):
        mask[i * hop : i * hop + frame] = True
    return mask


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    active_only: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按目标 SNR 混合语音与噪声，返回 ``(混合信号, 缩放后的噪声)``。

    **SNR 只在语音活跃段上定义**（``active_only=True``，DNS Challenge 的约定）。
    如果把静音段也算进语音功率，同一个"0 dB"在"话密"和"话稀"两段录音上
    实际信噪比能差 5 dB 以上，整个 SNR 维度的评测就没有可比性了。

    噪声不足时循环拼接，过长时随机截取一段。
    """
    rng = rng or np.random.default_rng()
    speech = np.asarray(speech, dtype=np.float64).reshape(-1)
    noise = np.asarray(noise, dtype=np.float64).reshape(-1)

    if noise.size < speech.size:
        noise = np.tile(noise, int(np.ceil(speech.size / noise.size)))
    if noise.size > speech.size:
        start = int(rng.integers(0, noise.size - speech.size + 1))
        noise = noise[start : start + speech.size]

    mask = speech_active_mask(speech) if active_only else np.ones(speech.size, dtype=bool)
    p_speech = float(np.mean(speech[mask] ** 2)) if mask.any() else float(np.mean(speech**2))
    p_noise = float(np.mean(noise**2))
    if p_noise <= 0 or p_speech <= 0:
        return speech.copy(), np.zeros_like(speech)

    scale = np.sqrt(p_speech / (p_noise * 10.0 ** (snr_db / 10.0)))
    noise_scaled = noise * scale
    return speech + noise_scaled, noise_scaled
