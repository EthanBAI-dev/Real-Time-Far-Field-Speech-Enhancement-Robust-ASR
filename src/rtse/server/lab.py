"""文件实验台的后端：一次性跑多种方法并给出完整指标对比。

它是整个项目的**离线主战场** —— 不需要麦克风、不需要训练好的模型就能用，
而且因为有干净参考信号，能算出实时演示里根本算不了的有参考指标
（SI-SDR / STOI / ESTOI）。

关键约束：**指标一律在内存数组上计算**，绝不写盘再读回来
（写盘的削波保护会改变幅度，污染 SDR、SegSNR 这类非尺度不变指标，见 ISSUES.md I-07）。
返回给前端的音频是单独编码的，与算指标的数组是同一份数据的两种用途。
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from rtse import SAMPLE_RATE
from rtse.data.synth import apply_rir, make_noise, make_rir, mix_at_snr
from rtse.dsp import build_dsp
from rtse.metrics.intrusive import estoi, pesq, seg_snr, sdr, si_sdr, stoi
from rtse.runtime import Pipeline
from rtse.vad import build_vad

__all__ = ["LabRequest", "run_lab"]

#: 实验台最长处理时长。再长的话一次跑 5 种方法会让请求超时，
#: 而且频谱图也看不过来。
MAX_SECONDS = 20.0


@dataclass
class LabRequest:
    audio: np.ndarray
    """干净参考信号（16 kHz 单声道）。"""
    noise_kind: str = "babble"
    snr_db: float = 5.0
    t60: float = 0.0
    methods: tuple[str, ...] = ("none", "specsub", "wiener", "mmse-lsa")
    vad: str = "energy"
    seed: int = 0


def _encode_wav(x: np.ndarray) -> str:
    """编码成 base64 的 WAV，供前端 <audio> 直接播放。

    这里做削波保护是**可以的**，因为指标已经在原始数组上算完了 ——
    这份编码结果只用于试听。
    """
    buf = io.BytesIO()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    y = x / peak if peak > 1.0 else x
    sf.write(buf, y, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _spectrogram(x: np.ndarray, n_bins: int = 96, n_cols: int = 400) -> list[list[float]]:
    """给前端画的静态频谱图：对数频率分组 + 时间轴抽稀到固定列数。"""
    from rtse.audio.stft import magnitude_db, stft
    from rtse.server.session import log_bin_edges

    spec = stft(x)
    if spec.shape[0] == 0:
        return []
    # 必须走共用的 magnitude_db —— 各处自己写 20*log10 是频谱图对不上的头号原因
    mag_db = magnitude_db(spec)

    groups = log_bin_edges(mag_db.shape[1], n_bins)
    binned = np.stack([mag_db[:, g].max(axis=1) for g in groups], axis=1)

    # 时间轴抽稀：取每段的最大值而不是抽样，避免漏掉短促的瞬态
    n_fr = binned.shape[0]
    if n_fr > n_cols:
        idx = np.linspace(0, n_fr, n_cols + 1).astype(int)
        binned = np.stack([binned[idx[i] : max(idx[i] + 1, idx[i + 1])].max(axis=0)
                           for i in range(n_cols)])
    return np.round(binned, 1).tolist()


def run_lab(req: LabRequest) -> dict:
    """执行一次实验，返回可直接 JSON 序列化的结果。"""
    rng = np.random.default_rng(req.seed)

    clean = np.asarray(req.audio, dtype=np.float64).reshape(-1)
    clean = clean[: int(MAX_SECONDS * SAMPLE_RATE)]
    peak = float(np.max(np.abs(clean)))
    if peak > 0:
        clean = clean / peak * 0.7

    # 先混响后加噪 —— 顺序不能反。真实场景里噪声和语音都经过房间，
    # 但噪声源通常离麦克风更近、更弥散，DNS Challenge 的约定就是
    # 语音卷 RIR 后再加噪。反过来做（先加噪再一起卷混响）会让噪声也带上
    # 与语音完全相同的混响特征，那是物理上不会发生的情况。
    reverb = apply_rir(clean, make_rir(req.t60, rng=rng)) if req.t60 > 0 else clean
    noise = make_noise(req.noise_kind, reverb.size, rng)
    noisy, _ = mix_at_snr(reverb, noise, req.snr_db, rng=rng)

    # 参考信号：有混响时用**混响后的干净语音**做参考，而不是原始干信号。
    # 否则降噪算法会因为"没能去掉混响"而被扣分，混淆了降噪和去混响两件事。
    reference = reverb

    rows = []
    for method in req.methods:
        enhancer = None if method in ("none", "noisy") else build_dsp(method)
        pipe = Pipeline(enhancer=enhancer, vad=build_vad(req.vad) if req.vad != "none" else None)
        enhanced, frames = pipe.process_signal(noisy)
        lat = pipe.latency_stats()

        rows.append(
            {
                "method": method,
                "metrics": {
                    "si_sdr": round(si_sdr(reference, enhanced), 2),
                    "sdr": round(sdr(reference, enhanced), 2),
                    "seg_snr": round(seg_snr(reference, enhanced), 2),
                    "stoi": round(stoi(reference, enhanced), 4),
                    "estoi": round(estoi(reference, enhanced), 4),
                    "pesq": (lambda v: round(v, 3) if v is not None else None)(
                        pesq(reference, enhanced)
                    ),
                },
                "latency": lat.as_dict(),
                "speech_ratio": round(float(np.mean([f.is_speech for f in frames])), 3),
                "audio": _encode_wav(enhanced),
                "spec": _spectrogram(enhanced),
            }
        )

    return {
        "config": {
            "noise_kind": req.noise_kind,
            "snr_db": req.snr_db,
            "t60": req.t60,
            "vad": req.vad,
            "duration_s": round(clean.size / SAMPLE_RATE, 2),
        },
        "reference": {"audio": _encode_wav(reference), "spec": _spectrogram(reference)},
        "noisy": {"audio": _encode_wav(noisy), "spec": _spectrogram(noisy)},
        "rows": rows,
        # PESQ 在 Windows 上装不上（ISSUES.md I-05），前端据此把该列标 n/a
        "pesq_available": pesq(reference, reference) is not None,
    }
