"""V1 三测试集生成逻辑的最小本地验证。"""

import json

import numpy as np

from rtse.audio.io import read_audio, write_audio
from rtse.data.benchmarks import (
    BenchmarkSource,
    generate_controlled_benchmark,
    generate_real_cer_benchmark,
)
from rtse.data.synth import speech_active_mask


def _tone(seconds=1.2, freq=220.0):
    t = np.arange(int(16000 * seconds)) / 16000
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_controlled_schema_keeps_rt60_and_task_boundaries(tmp_path):
    noise = tmp_path / "noise.wav"
    rir = tmp_path / "rir.wav"
    # 故意加入很大的 DC 偏置，复现 I-32：旧实现的名义/实测 SNR 会偏数 dB。
    biased_noise = np.random.default_rng(0).normal(0, 0.05, 16000 * 2) + 0.4
    write_audio(noise, biased_noise)
    impulse = np.zeros(8000)
    impulse[0] = 1.0
    impulse[100] = 0.2
    write_audio(rir, impulse)

    real = {t: [(rir, t)] for t in (0.2, 0.4, 0.6, 0.8)}
    idx = generate_controlled_benchmark(
        [BenchmarkSource("s1", _tone(), "测试文本")],
        {"stationary": [noise], "nonstationary": [noise]},
        real,
        tmp_path / "controlled",
        dataset_id="unit",
        purpose="controlled_chinese_cer",
        source_dataset="unit",
        per_cell=1,
        snrs=(0,),
    )

    assert len(idx["records"]) == 2 * 2 * 4 + 1
    assert idx["reference_is_clean"] is True
    assert idx["cer_upper_is_meaningful"] is True
    assert idx["target_definition"] == "reverberant_noise_free"
    assert idx["strata"][-1] == "rt60_bucket"
    mixed = [r for r in idx["records"] if r["noise_kind"] != "none"]
    assert {r["rt60_bucket"] for r in mixed} == {0.2, 0.4, 0.6, 0.8}
    assert {r["rir_kind"] for r in idx["records"]} == {"synth", "real", "none"}
    identity = [r for r in idx["records"] if r["noise_kind"] == "none"]
    assert len(identity) == 1 and identity[0]["snr"] == "clean"
    for record in mixed:
        assert abs(record["snr_measured"] - record["snr"]) < 0.05
        target = read_audio(tmp_path / "controlled" / record["clean"])
        noisy = read_audio(tmp_path / "controlled" / record["noisy"])
        residual = noisy - target
        signal = target - target.mean()
        noise_ac = residual - residual.mean()
        mask = speech_active_mask(signal)
        # 写盘后的输入/目标必须仍保持同一个 SNR，不能各自归一化。
        actual = 10 * np.log10(np.mean(signal[mask] ** 2) / np.mean(noise_ac**2))
        assert abs(actual - record["snr"]) < 0.15


def test_real_cer_set_has_no_fake_clean_upper(tmp_path):
    sources = [
        BenchmarkSource(f"s{i}", _tone(seconds), f"文本{i}")
        for i, seconds in enumerate((3.0, 6.0, 10.0))
    ]
    idx = generate_real_cer_benchmark(sources, tmp_path / "real", per_duration_bucket=1)

    assert idx["reference_is_clean"] is False
    assert idx["cer_upper_is_meaningful"] is False
    assert len(idx["records"]) == 3
    assert all("clean" not in r for r in idx["records"])
    on_disk = json.loads((tmp_path / "real" / "index.json").read_text(encoding="utf-8"))
    assert on_disk["strata"] == ["duration_bucket"]
