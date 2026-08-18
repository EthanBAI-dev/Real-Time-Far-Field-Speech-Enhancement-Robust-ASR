"""V1 的三套评测集生成器。

这个模块只负责固定混音与索引 schema；下载和 parquet 解码留在 Colab
notebook。生成逻辑因此能在本地用微型数据验证，不必等大语料下载完成。

V1 明确只做单通道去噪：输入语音可以带混响，但有参考指标比较的是
``带混响、无加性噪声``目标。模型没有被要求去混响。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rtse import SAMPLE_RATE
from rtse.audio.io import read_audio, write_audio
from rtse.data.synth import apply_rir, make_rir, mix_at_snr, speech_active_mask
from rtse.dsp.rt60 import estimate_t60

__all__ = [
    "RT60_BINS",
    "BenchmarkSource",
    "build_real_rir_buckets",
    "generate_controlled_benchmark",
    "generate_real_cer_benchmark",
]


@dataclass(frozen=True)
class BenchmarkSource:
    """一条待合成的源语音。``text=None`` 表示只做声学质量评测。"""

    source_id: str
    audio: np.ndarray
    text: str | None = None


# 真实 RIR 先按实测 RT60 分桶，再与同一目标档的合成 RIR 比较。
RT60_BINS: tuple[tuple[float, float, float], ...] = (
    (0.2, 0.10, 0.30),
    (0.4, 0.30, 0.50),
    (0.6, 0.50, 0.70),
    (0.8, 0.70, 1.00),
)


def _bucket_for_rt60(value: float) -> float | None:
    for target, lo, hi in RT60_BINS:
        if lo <= value < hi:
            return target
    return None


def build_real_rir_buckets(
    rir_paths: Iterable[str | Path],
    *,
    min_per_bucket: int = 20,
    max_scan: int = 8000,
    seed: int = 20260818,
) -> dict[float, list[tuple[str, float]]]:
    """把真实 RIR 按实测 RT60 分到 0.2/0.4/0.6/0.8 秒四档。

    四档没收齐就报错，禁止悄悄退回“随便抽一个真实 RIR”——那正是旧测试
    中真实 RIR 中位数 2.20 秒所造成的混淆变量。
    """

    paths = [str(p) for p in rir_paths]
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    buckets: dict[float, list[tuple[str, float]]] = {t: [] for t, _, _ in RT60_BINS}
    for path in paths[:max_scan]:
        try:
            measured = float(estimate_t60(read_audio(path)))
        except Exception:  # noqa: BLE001, S112 - 真实数据中的坏文件只跳过
            continue
        if not np.isfinite(measured):
            continue
        target = _bucket_for_rt60(measured)
        if target is not None and len(buckets[target]) < min_per_bucket:
            buckets[target].append((path, measured))
        if all(len(v) >= min_per_bucket for v in buckets.values()):
            break

    missing = {t: len(v) for t, v in buckets.items() if len(v) < min_per_bucket}
    if missing:
        raise RuntimeError(
            f"真实 RIR 的 RT60 匹配桶不够：{missing}；已扫描最多 {min(max_scan, len(paths))} 条。"
            "请调大 max_scan，不要用超强混响 RIR 顶替。"
        )
    return buckets


def _valid_source(source: BenchmarkSource, *, normalize: bool = True) -> BenchmarkSource:
    audio = np.asarray(source.audio, dtype=np.float64).reshape(-1)
    if audio.size < SAMPLE_RATE or not np.all(np.isfinite(audio)):
        raise ValueError(f"源语音 {source.source_id!r} 太短或包含非有限值")
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-9:
        raise ValueError(f"源语音 {source.source_id!r} 是静音")
    if normalize:
        audio = audio / peak * 0.7
    return BenchmarkSource(source.source_id, audio, source.text)


def _take_noise(paths: Sequence[str | Path], n: int, rng: np.random.Generator) -> np.ndarray:
    if not paths:
        raise ValueError("噪声池为空")
    for _ in range(100):
        try:
            y = read_audio(paths[int(rng.integers(len(paths)))])
        except Exception:  # noqa: BLE001, S112
            continue
        if y.size < SAMPLE_RATE // 2 or not np.all(np.isfinite(y)):
            continue
        if y.size < n:
            y = np.tile(y, int(np.ceil(n / y.size)))
        start = int(rng.integers(y.size - n + 1)) if y.size > n else 0
        seg = np.asarray(y[start : start + n], dtype=np.float64)
        if seg.size == n and np.mean((seg - seg.mean()) ** 2) > 1e-12:
            return seg
    raise RuntimeError("连续 100 次没有取到有交流能量的噪声片段")


def _measured_snr(clean: np.ndarray, scaled_noise: np.ndarray) -> float:
    signal = clean - clean.mean()
    noise = scaled_noise - scaled_noise.mean()
    mask = speech_active_mask(signal)
    pn = float(np.mean(noise**2))
    ps = float(np.mean(signal[mask] ** 2))
    if not np.isfinite(ps) or not np.isfinite(pn) or min(ps, pn) <= 0:
        raise ValueError("无法计算有限的实测 SNR")
    return float(10.0 * np.log10(ps / pn))


def _write_aligned_pair(
    input_path: Path,
    target_path: Path,
    noisy: np.ndarray,
    target: np.ndarray,
) -> float:
    """用同一增益写入输入/目标，避免独立削波保护破坏配对关系。"""

    peak = max(float(np.max(np.abs(noisy))), float(np.max(np.abs(target))))
    gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0
    write_audio(input_path, noisy * gain, guard_clipping=False)
    write_audio(target_path, target * gain, guard_clipping=False)
    return gain


def generate_controlled_benchmark(
    sources: Sequence[BenchmarkSource],
    noise_by_kind: Mapping[str, Sequence[str | Path]],
    real_rir_buckets: Mapping[float, Sequence[tuple[str | Path, float]]],
    output_dir: str | Path,
    *,
    dataset_id: str,
    purpose: str,
    source_dataset: str,
    per_cell: int,
    snrs: Sequence[int] = (-5, 0, 5, 10, 15),
    seed: int = 20260818,
) -> dict:
    """生成受控去噪测试集。

    完整分层是 SNR × 噪声平稳性 × RIR 来源 × RT60 桶。真实与合成 RIR
    使用相同 RT60 桶；参考始终是 ``wet``（有混响、无加性噪声）。
    """

    if per_cell <= 0:
        raise ValueError("per_cell 必须大于 0")
    prepared = [_valid_source(s) for s in sources]
    if not prepared:
        raise ValueError("源语音为空")
    for kind in ("stationary", "nonstationary"):
        if not noise_by_kind.get(kind):
            raise ValueError(f"缺少 {kind} 噪声")
    for target, _, _ in RT60_BINS:
        if not real_rir_buckets.get(target):
            raise ValueError(f"真实 RIR 的 {target}s 桶为空")

    out = Path(output_dir)
    audio_dir = out / "audio"
    if audio_dir.exists():
        for p in audio_dir.glob("*.wav"):
            p.unlink()
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    cells = [
        (int(snr), noise_kind, rir_kind, target)
        for snr in snrs
        for noise_kind in ("stationary", "nonstationary")
        for rir_kind in ("synth", "real")
        for target, _, _ in RT60_BINS
    ]
    records: list[dict] = []
    for cell_i, (snr, noise_kind, rir_kind, target_rt60) in enumerate(cells):
        for k in range(per_cell):
            source = prepared[(cell_i * per_cell + k) % len(prepared)]
            dry = source.audio
            if rir_kind == "synth":
                rir = make_rir(target_rt60, rng=rng)
                measured_rt60 = float(estimate_t60(rir))
            else:
                pool = real_rir_buckets[target_rt60]
                rir_path, measured_rt60 = pool[int(rng.integers(len(pool)))]
                rir = read_audio(rir_path)
            wet = apply_rir(dry, rir)
            # RIR 的绝对增益没有统一物理含义；先把去噪目标统一到安全峰值，
            # 再按 SNR 加噪。后续若仍需防削波，输入和目标必须共用同一个增益。
            wet_peak = float(np.max(np.abs(wet)))
            if wet_peak <= 1e-9:
                raise ValueError(f"源语音 {source.source_id!r} 与 RIR 卷积后近似静音")
            wet = wet / wet_peak * 0.7
            noise = _take_noise(noise_by_kind[noise_kind], wet.size, rng)
            noisy, scaled_noise = mix_at_snr(wet, noise, snr, rng=rng)
            measured_snr = _measured_snr(wet, scaled_noise)

            stem = f"{len(records):05d}_{noise_kind}_snr{snr}_{rir_kind}_rt{target_rt60:.1f}"
            storage_gain = _write_aligned_pair(
                audio_dir / f"{stem}_input.wav",
                audio_dir / f"{stem}_target.wav",
                noisy,
                wet,
            )
            record = {
                "id": stem,
                "source_id": source.source_id,
                "noisy": f"audio/{stem}_input.wav",
                "clean": f"audio/{stem}_target.wav",
                "duration_s": round(wet.size / SAMPLE_RATE, 3),
                "snr": snr,
                "snr_measured": round(measured_snr, 2),
                "storage_gain": round(storage_gain, 8),
                "noise_kind": noise_kind,
                "rir_kind": rir_kind,
                "rt60_bucket": target_rt60,
                "rt60_nominal": target_rt60 if rir_kind == "synth" else None,
                "rt60_measured": round(float(measured_rt60), 3),
            }
            if source.text:
                record["text"] = source.text
            records.append(record)

    # 独立的 clean→clean 实验格：旧模型最严重的失败就是对完全干净输入仍然下重手。
    # 不能只靠 15 dB 近似，也不能假设模型会自动学会透传。
    for k in range(per_cell):
        source = prepared[(len(cells) * per_cell + k) % len(prepared)]
        stem = f"{len(records):05d}_identity_clean"
        write_audio(audio_dir / f"{stem}_input.wav", source.audio)
        write_audio(audio_dir / f"{stem}_target.wav", source.audio)
        record = {
            "id": stem,
            "source_id": source.source_id,
            "noisy": f"audio/{stem}_input.wav",
            "clean": f"audio/{stem}_target.wav",
            "duration_s": round(source.audio.size / SAMPLE_RATE, 3),
            "snr": "clean",
            "snr_measured": None,
            "noise_kind": "none",
            "rir_kind": "none",
            "rt60_bucket": 0.0,
            "rt60_nominal": 0.0,
            "rt60_measured": 0.0,
        }
        if source.text:
            record["text"] = source.text
        records.append(record)

    has_text = all(bool(r.get("text")) for r in records)
    index = {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "purpose": purpose,
        "source_dataset": source_dataset,
        "sample_rate": SAMPLE_RATE,
        "target_definition": "reverberant_noise_free",
        "reference_is_clean": True,
        "reference_note": (
            "clean 字段表示去噪任务的无加性噪声目标；它可以带混响。"
            "V1 不把未去除混响计为降噪错误。"
        ),
        "cer_upper_is_meaningful": has_text,
        "strata": ["snr", "noise_kind", "rir_kind", "rt60_bucket"],
        "cer_per_cell_default": per_cell,
        "per_cell": per_cell,
        "snrs": list(snrs),
        "rt60_buckets": [t for t, _, _ in RT60_BINS],
        "has_identity_cell": True,
        "records": records,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return index


def generate_real_cer_benchmark(
    sources: Sequence[BenchmarkSource],
    output_dir: str | Path,
    *,
    per_duration_bucket: int,
    dataset_id: str = "wenetspeech_meeting_real",
) -> dict:
    """保存不再二次加噪/卷 RIR 的真实会议 CER 集。"""

    if per_duration_bucket <= 0:
        raise ValueError("per_duration_bucket 必须大于 0")
    buckets: dict[str, list[BenchmarkSource]] = {"short": [], "medium": [], "long": []}
    for raw in sources:
        # 真实场景集不做增益归一化；否则就不再是“原始会议输入直接增强”。
        source = _valid_source(raw, normalize=False)
        if not source.text:
            continue
        seconds = source.audio.size / SAMPLE_RATE
        label = "short" if seconds < 5.0 else "medium" if seconds < 9.0 else "long"
        if len(buckets[label]) < per_duration_bucket:
            buckets[label].append(source)
    missing = {k: len(v) for k, v in buckets.items() if len(v) < per_duration_bucket}
    if missing:
        raise ValueError(f"真实会议语音时长分桶不足：{missing}")

    out = Path(output_dir)
    audio_dir = out / "audio"
    if audio_dir.exists():
        for p in audio_dir.glob("*.wav"):
            p.unlink()
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for bucket, group in buckets.items():
        for source in group:
            stem = f"{len(records):05d}_{bucket}_{source.source_id}"
            write_audio(audio_dir / f"{stem}_input.wav", source.audio)
            records.append({
                "id": stem,
                "source_id": source.source_id,
                "noisy": f"audio/{stem}_input.wav",
                "text": source.text,
                "duration_s": round(source.audio.size / SAMPLE_RATE, 3),
                "duration_bucket": bucket,
            })

    index = {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "purpose": "real_world_cer",
        "source_dataset": "WenetSpeech test_meeting",
        "sample_rate": SAMPLE_RATE,
        "reference_is_clean": False,
        "reference_note": "真实会议录音本身含噪与混响，只比较输入与增强后的 CER。",
        "cer_upper_is_meaningful": False,
        "strata": ["duration_bucket"],
        "cer_per_cell_default": per_duration_bucket,
        "records": records,
    }
    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return index
