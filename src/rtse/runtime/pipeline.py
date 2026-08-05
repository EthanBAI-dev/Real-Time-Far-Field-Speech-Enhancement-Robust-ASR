"""实时管线：把 VAD、增强器、计量串成一条逐帧处理的流水线。

这是 Web 演示和 RTF/延迟测量共用的同一段代码 —— 刻意如此。
如果演示走一条路径、测量走另一条，测出来的数字就跟用户实际听到的东西无关了。

延迟的三个层次，报告里必须分开讲，否则"延迟 32 ms"这句话没有意义：

1. **算法延迟**（``algorithmic_latency_ms``）：由 STFT 窗长决定，32 ms。
   即使 CPU 无限快也消除不掉，是方法本身的固有延迟。
2. **处理延迟**（``processing`` 分位数）：每帧真实计算耗时。
   它必须 **< 帧移（16 ms）**，否则缓冲区会持续累积，延迟无界增长直至崩溃。
3. **端到端延迟**：算法延迟 + 处理延迟 + 缓冲/传输。Web 演示里还要加上
   浏览器 AudioWorklet 的缓冲和 WebSocket 往返，这部分单独测。

RTF（实时系数）= 处理耗时 / 音频时长。RTF < 1 才可能实时，
但实际部署要留余量，一般要求 RTF < 0.5。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from rtse import SAMPLE_RATE
from rtse.audio.stft import (
    DEFAULT_CONFIG,
    STFTConfig,
    StreamingISTFT,
    StreamingSTFT,
    magnitude_db,
    num_frames,
)
from rtse.dsp.base import PassThrough, StreamingEnhancer
from rtse.vad.base import StreamingVAD

__all__ = ["FrameResult", "LatencyStats", "Pipeline"]


@dataclass
class FrameResult:
    """单帧的处理结果。Web 端逐帧推送它的一个精简版本。"""

    samples: np.ndarray
    """增强后的 ``hop`` 个时域样本。"""
    is_speech: bool
    vad_prob: float
    noisy_mag_db: np.ndarray
    """带噪幅度谱（dB），可视化用。"""
    enhanced_mag_db: np.ndarray
    gain_db: np.ndarray | None
    """本帧施加的增益（dB）。神经模型可给出等效增益，DSP 直接给。"""
    proc_ms: float
    """本帧的纯计算耗时（毫秒），不含 I/O。"""


@dataclass
class LatencyStats:
    """逐帧耗时统计。分位数比均值重要得多 ——
    实时系统的失败是由**尾部**决定的：p99 超过帧移就会周期性爆音，
    而均值可能只有帧移的十分之一，看上去毫无问题。"""

    n_frames: int
    audio_seconds: float
    total_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    frame_budget_ms: float
    algorithmic_latency_ms: float

    @property
    def rtf(self) -> float:
        return self.total_ms / 1000.0 / self.audio_seconds if self.audio_seconds > 0 else float("nan")

    @property
    def realtime_ok(self) -> bool:
        """p99 是否守住了帧预算。这是"能不能实时"的判据，不是 RTF 均值。"""
        return self.p99_ms < self.frame_budget_ms

    def as_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "audio_seconds": round(self.audio_seconds, 3),
            "rtf": round(self.rtf, 5),
            "proc_p50_ms": round(self.p50_ms, 3),
            "proc_p95_ms": round(self.p95_ms, 3),
            "proc_p99_ms": round(self.p99_ms, 3),
            "proc_max_ms": round(self.max_ms, 3),
            "frame_budget_ms": round(self.frame_budget_ms, 3),
            "algorithmic_latency_ms": round(self.algorithmic_latency_ms, 3),
            "realtime_ok": self.realtime_ok,
        }


class Pipeline:
    """VAD + 增强的逐帧流水线。

    Args:
        enhancer: 增强器。None 表示直通（``noisy`` 基线）。
        vad: VAD。None 表示不做检测（全部帧视为语音）。
        vad_gate: 是否让 VAD 真的**门控**输出。
            默认 False —— VAD 结果只用于展示和送 ASR 分段，**不**静音非语音段。
            设 True 时非语音段会被平滑地压低到 ``gate_floor_db``。
            这个开关本身就是一个实验：门控能进一步降低背景噪声，
            但一旦 VAD 误切，代价是直接丢字。
        gate_attack_ms: 门控**打开**的时间常数。必须短 —— 语音起始被削掉
            就是直接丢字。
        gate_release_ms: 门控**关闭**的时间常数。必须长 —— 它同时充当了
            "软挂起"，能吸收 VAD 在音节间隙的短暂翻转。

    关于门控平滑：直接按帧硬切（gain 在 1.0 和 0.1 之间瞬间跳变）有两个问题：
    一是每次跳变都是一个阶跃，听感上是"咔哒"声；二是 VAD 在音节间隙难免抖动，
    硬切会把连续语音削得坑坑洼洼。所以这里用非对称的一阶平滑：**快开慢关**。
    """

    def __init__(
        self,
        enhancer: StreamingEnhancer | None = None,
        vad: StreamingVAD | None = None,
        cfg: STFTConfig = DEFAULT_CONFIG,
        vad_gate: bool = False,
        gate_floor_db: float = -20.0,
        gate_attack_ms: float = 5.0,
        gate_release_ms: float = 200.0,
    ) -> None:
        self.cfg = cfg
        self.enhancer = enhancer or PassThrough(cfg)
        self.vad = vad
        self.vad_gate = vad_gate
        self.gate_floor = 10.0 ** (gate_floor_db / 20.0)
        frame_s = cfg.hop / SAMPLE_RATE
        self._gate_a_atk = float(np.exp(-frame_s / max(gate_attack_ms / 1000.0, 1e-6)))
        self._gate_a_rel = float(np.exp(-frame_s / max(gate_release_ms / 1000.0, 1e-6)))
        self._stft = StreamingSTFT(cfg)
        self._istft = StreamingISTFT(cfg)
        self._times: list[float] = []
        self.reset()

    @property
    def frame_budget_ms(self) -> float:
        """每帧的计算预算：一个帧移的时长。超过它就跟不上实时。"""
        return self.cfg.hop / SAMPLE_RATE * 1000.0

    def reset(self) -> None:
        self._stft.reset()
        self._istft.reset()
        self.enhancer.reset()
        if self.vad is not None:
            self.vad.reset()
        self._times = []
        self._gate = 1.0  # 从全开始，避免开头第一个字被门控吃掉

    def process_block(self, block: np.ndarray) -> FrameResult:
        """处理一个 ``hop`` 长的时域块。这是实时路径上唯一被调用的函数。"""
        t0 = time.perf_counter()

        spec = self._stft.push(block)

        if self.vad is not None:
            vf = self.vad.process(block, spec)
            is_speech, vad_prob = vf.is_speech, vf.prob
        else:
            is_speech, vad_prob = True, 1.0

        enh_spec = self.enhancer.process_frame(spec)
        if self.vad_gate:
            # 非对称一阶平滑：快开慢关。慢关顺带充当软挂起，
            # 吸收 VAD 在音节间隙的短暂翻转，避免把连续语音削出坑。
            target = 1.0 if is_speech else self.gate_floor
            a = self._gate_a_atk if target > self._gate else self._gate_a_rel
            self._gate = a * self._gate + (1.0 - a) * target
            enh_spec = enh_spec * self._gate

        samples = self._istft.push(enh_spec)
        proc_ms = (time.perf_counter() - t0) * 1000.0
        self._times.append(proc_ms)

        gain = self.enhancer.last_gain
        return FrameResult(
            samples=samples,
            is_speech=is_speech,
            vad_prob=vad_prob,
            noisy_mag_db=magnitude_db(spec, self.cfg),
            enhanced_mag_db=magnitude_db(enh_spec, self.cfg),
            gain_db=20.0 * np.log10(np.maximum(gain, 1e-6)) if gain is not None else None,
            proc_ms=proc_ms,
        )

    def process_signal(self, x: np.ndarray) -> tuple[np.ndarray, list[FrameResult]]:
        """整段处理，返回 ``(增强信号, 逐帧结果)``。走的是与实时完全相同的路径。"""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        n_fr = num_frames(x.size, self.cfg)
        feed = np.zeros(n_fr * self.cfg.hop)
        feed[: x.size] = x

        self.reset()
        frames = [
            self.process_block(feed[i * self.cfg.hop : (i + 1) * self.cfg.hop]) for i in range(n_fr)
        ]
        out = np.concatenate([f.samples for f in frames])
        return out[self.cfg.pad : self.cfg.pad + x.size], frames

    def latency_stats(self) -> LatencyStats:
        t = np.asarray(self._times) if self._times else np.zeros(1)
        return LatencyStats(
            n_frames=len(self._times),
            audio_seconds=len(self._times) * self.cfg.hop / SAMPLE_RATE,
            total_ms=float(t.sum()),
            p50_ms=float(np.percentile(t, 50)),
            p95_ms=float(np.percentile(t, 95)),
            p99_ms=float(np.percentile(t, 99)),
            max_ms=float(t.max()),
            frame_budget_ms=self.frame_budget_ms,
            algorithmic_latency_ms=self.cfg.latency_samples / SAMPLE_RATE * 1000.0,
        )
