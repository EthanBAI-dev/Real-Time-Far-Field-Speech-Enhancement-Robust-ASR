"""每个 WebSocket 连接的会话状态。

一个会话持有一条 ``Pipeline``。热切换方法时**重建**管线而不是复用 ——
噪声估计器、决策导向的历史增益、VAD 的本底跟踪都是有状态的，
复用会让新方法继承上一个方法的内部状态，产生几秒钟的诡异过渡。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from rtse import HOP_LENGTH, N_FREQ, SAMPLE_RATE
from rtse.dsp import build_dsp
from rtse.runtime import Pipeline
from rtse.vad import build_vad

__all__ = ["SessionConfig", "StreamSession", "SPEC_BINS", "log_bin_edges"]

#: 送到前端的频谱条数。257 个 bin 全送是浪费（画布也就几百像素宽），
#: 而且 62.5 fps × 257 × 2 路的 JSON 序列化开销会真的吃掉 CPU 预算。
SPEC_BINS = 64


def log_bin_edges(n_freq: int = N_FREQ, n_bins: int = SPEC_BINS) -> list[np.ndarray]:
    """对数间隔的频率分组索引。

    用对数而不是线性分组：线性分组下 0~1 kHz（承载基频和第一共振峰、
    信息量最大的一段）只占 64 条里的 4 条，肉眼几乎看不出变化。
    对数分组后低频占到约一半，频谱图上才能看清降噪对语音谐波做了什么。
    """
    freqs = np.fft.rfftfreq((n_freq - 1) * 2, 1.0 / SAMPLE_RATE)
    lo, hi = 60.0, SAMPLE_RATE / 2
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    groups = []
    for i in range(n_bins):
        idx = np.flatnonzero((freqs >= edges[i]) & (freqs < edges[i + 1]))
        if idx.size == 0:  # 低频段可能一个 bin 都落不进来，取最近的
            idx = np.array([int(np.argmin(np.abs(freqs - edges[i])))])
        groups.append(idx)
    return groups


_BIN_GROUPS = log_bin_edges()


def _downsample_spec(mag_db: np.ndarray) -> list[float]:
    """按对数分组取每组最大值。取最大而不是平均 ——
    平均会把窄带谐波峰抹平，而谐波结构正是我们想让用户看见的东西。"""
    return [round(float(mag_db[g].max()), 1) for g in _BIN_GROUPS]


@dataclass
class SessionConfig:
    method: str = "wiener"
    """增强方法：``none`` / ``specsub`` / ``wiener`` / ``mmse-lsa`` / 神经模型名。"""
    vad: str = "energy"
    """``none`` / ``energy`` / ``webrtc``。"""
    vad_gate: bool = False
    asr: bool = False


@dataclass
class ChunkResult:
    """一个音频块的处理结果，序列化后发给前端。"""

    seq: int
    audio: np.ndarray
    meta: dict = field(default_factory=dict)


class StreamSession:
    """一路实时音频流的处理会话。"""

    def __init__(self, cfg: SessionConfig | None = None) -> None:
        self.cfg = cfg or SessionConfig()
        self._pipeline: Pipeline | None = None
        self._residual = np.zeros(0, dtype=np.float64)
        self._seq = 0
        self._proc_ms_hist: list[float] = []
        self._speech_buf: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self.rebuild()

    # ------------------------------------------------------------------ 配置

    def rebuild(self) -> None:
        """按当前配置重建管线。方法切换时调用。"""
        from rtse.dsp import DSP_METHODS
        from rtse.runtime import OnnxEnhancer, discover_onnx_models

        method = self.cfg.method
        if method in ("none", "noisy"):
            enh = None
        elif method in DSP_METHODS:
            enh = build_dsp(method)
        else:
            # 神经模型：名字对应 models/<name>.onnx
            models = discover_onnx_models()
            if method not in models:
                raise KeyError(f"未知方法 {method!r}；可用神经模型：{sorted(models)}")
            enh = OnnxEnhancer(models[method])
        vad = None if self.cfg.vad == "none" else build_vad(self.cfg.vad)
        self._pipeline = Pipeline(enhancer=enh, vad=vad, vad_gate=self.cfg.vad_gate)
        self._residual = np.zeros(0, dtype=np.float64)
        self._speech_buf.clear()
        self._speech_run = self._silence_run = 0

    def update_config(self, **kwargs) -> None:
        changed = False
        for k, v in kwargs.items():
            if hasattr(self.cfg, k) and getattr(self.cfg, k) != v:
                setattr(self.cfg, k, v)
                changed = True
        if changed:
            self.rebuild()

    # ------------------------------------------------------------------ 处理

    def process(self, pcm: np.ndarray) -> ChunkResult:
        """处理任意长度的输入块。

        输入长度不必是 ``hop`` 的整数倍 —— 浏览器送来的块大小受
        AudioWorklet 缓冲和采样率转换影响，不保证对齐。
        不足一帧的尾巴留在 ``_residual`` 里，下次拼上。
        """
        assert self._pipeline is not None
        t0 = time.perf_counter()

        buf = np.concatenate([self._residual, np.asarray(pcm, dtype=np.float64).reshape(-1)])
        n_fr = buf.size // HOP_LENGTH
        self._residual = buf[n_fr * HOP_LENGTH :]

        frames = [self._pipeline.process_block(buf[i * HOP_LENGTH : (i + 1) * HOP_LENGTH])
                  for i in range(n_fr)]

        wall_ms = (time.perf_counter() - t0) * 1000.0
        audio = (np.concatenate([f.samples for f in frames]) if frames
                 else np.zeros(0, dtype=np.float64))

        self._proc_ms_hist.extend(f.proc_ms for f in frames)
        del self._proc_ms_hist[:-400]  # 只保留最近约 6.4 秒

        self._seq += 1
        meta = {
            "seq": self._seq,
            "n_frames": n_fr,
            "vad": [round(f.vad_prob, 3) for f in frames],
            "speech": [bool(f.is_speech) for f in frames],
            "noisy_spec": [_downsample_spec(f.noisy_mag_db) for f in frames],
            "enh_spec": [_downsample_spec(f.enhanced_mag_db) for f in frames],
            **self._stats(frames, buf[: n_fr * HOP_LENGTH], audio, wall_ms),
        }
        return ChunkResult(seq=self._seq, audio=audio, meta=meta)

    def _stats(self, frames, input_pcm: np.ndarray, out: np.ndarray, wall_ms: float) -> dict:
        hist = np.asarray(self._proc_ms_hist) if self._proc_ms_hist else np.zeros(1)
        audio_ms = len(frames) * HOP_LENGTH / SAMPLE_RATE * 1000.0
        return {
            "in_db": round(_rms_db(input_pcm), 1),
            "out_db": round(_rms_db(out), 1),
            "proc_ms": round(wall_ms, 2),
            # RTF 用真实墙钟时间（含序列化前的全部处理），不是逐帧耗时之和
            "rtf": round(wall_ms / audio_ms, 4) if audio_ms > 0 else 0.0,
            "p50_ms": round(float(np.percentile(hist, 50)), 2),
            "p95_ms": round(float(np.percentile(hist, 95)), 2),
            "p99_ms": round(float(np.percentile(hist, 99)), 2),
            "budget_ms": round(HOP_LENGTH / SAMPLE_RATE * 1000.0, 2),
            "algo_latency_ms": round(self._pipeline.cfg.latency_samples / SAMPLE_RATE * 1000.0, 1),
        }

    # ------------------------------------------------------- ASR 分段（供后续接入）

    def collect_speech(self, frames, audio: np.ndarray) -> np.ndarray | None:
        """按 VAD 累积语音段，静音足够长时吐出一整段供 ASR 识别。

        这是 VAD 在本项目里的**主要用途** —— 不是为了静音降噪，
        而是为了给流式 ASR 切出合理的识别单元。切得太碎，
        识别器拿不到足够的上下文；切得太粗，字幕延迟就上去了。
        """
        if not frames:
            return None
        speech_now = any(f.is_speech for f in frames)
        if speech_now:
            self._speech_run += len(frames)
            self._silence_run = 0
            self._speech_buf.append(audio)
        else:
            self._silence_run += len(frames)
            if self._speech_buf:
                self._speech_buf.append(audio)  # 尾部静音一并保留，别切掉尾音

        # 静音持续 ~0.5 s 且已积累 ≥0.6 s 语音 → 出段
        if self._silence_run >= 30 and self._speech_run >= 38 and self._speech_buf:
            seg = np.concatenate(self._speech_buf)
            self._speech_buf.clear()
            self._speech_run = self._silence_run = 0
            return seg
        # 超过 12 秒强制出段，避免连续说话时字幕永远不出来
        if sum(b.size for b in self._speech_buf) > SAMPLE_RATE * 12:
            seg = np.concatenate(self._speech_buf)
            self._speech_buf.clear()
            self._speech_run = self._silence_run = 0
            return seg
        return None


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -100.0
    return float(20.0 * np.log10(np.sqrt(np.mean(x**2)) + 1e-10))
