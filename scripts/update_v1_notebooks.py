"""把三个 Colab notebook 同步到 V1 三测试集设计。

Notebook 是 JSON，手工同时改三份共享配置很容易漂移。这个脚本只替换明确的
标题/数据生成 cell，并清空旧输出；下载与解压的硬化代码保持原样。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def _lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def _load(name: str) -> dict:
    return json.loads((NB_DIR / name).read_text(encoding="utf-8"))


def _save(name: str, nb: dict) -> None:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    nb.get("metadata", {}).pop("widgets", None)
    (NB_DIR / name).write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def _set(nb: dict, index: int, kind: str, text: str) -> None:
    cell = nb["cells"][index]
    if cell["cell_type"] != kind:
        raise RuntimeError(f"cell {index} 预期 {kind}，实际 {cell['cell_type']}")
    cell["source"] = _lines(text)


def _new_cell(kind: str, text: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": _lines(text)}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _update_common_config(nb: dict) -> None:
    src = "".join(nb["cells"][2]["source"])
    src = src.replace(
        "# True = 跳过 DNS 大文件，只下 WenetSpeech（约 520 MB）验证整条链路。",
        "# V1 已停用 QUICK_TEST：两套受控集都依赖 DNS 留出噪声/RIR。\n"
        "# 小规模跑通请用 SMOKE_RUN=True；QUICK_TEST 必须保持 False。",
    )
    src = src.replace(
        "# True = 只下载两份中文 parquet 并生成 WenetSpeech 真实 CER 小集；不训练。\n"
        "# 要验证训练闭环，请保持 False，并使用 SMOKE_RUN=True。",
        "# V1 已停用 QUICK_TEST：两套受控集都依赖 DNS 留出噪声/RIR。\n"
        "# 小规模跑通请用 SMOKE_RUN=True；QUICK_TEST 必须保持 False。",
    )
    src = src.replace(
        "#   测试集   50 格 × 3 = 150 条（正式 15/格 = 750）",
        "#   受控集   81 格 × 1 = 81 条/套（正式 5/格 = 405）\n"
        "#   真实 CER 3 个时长桶 × 10 = 30 条（正式 300）",
    )
    src = src.replace(
        "#   受控集   80 格 × 1 = 80 条/套（正式 5/格 = 400）",
        "#   受控集   81 格 × 1 = 81 条/套（正式 5/格 = 405）",
    )
    src = src.replace(
        "TESTSET_DIR = f'{DRIVE}/testset'    # 固定测试集",
        "TESTSETS_DIR = f'{DRIVE}/testsets'  # V1 三套职责分离的固定测试集\n"
        "DNS_QUALITY_DIR = f'{TESTSETS_DIR}/dns_objective'\n"
        "AISHELL_CER_DIR = f'{TESTSETS_DIR}/aishell_controlled'\n"
        "WENET_REAL_DIR = f'{TESTSETS_DIR}/wenetspeech_real'",
    )
    src = src.replace(
        "for d in [WORK, ARCHIVE_DIR, DATA, CKPT_DIR, MODEL_DIR, TESTSET_DIR, LOG_DIR]:",
        "for d in [WORK, ARCHIVE_DIR, DATA, CKPT_DIR, MODEL_DIR, TESTSETS_DIR, LOG_DIR]:",
    )
    src = src.replace(
        "print(f'  固定测试集          {TESTSET_DIR}')",
        "print(f'  三套固定测试集      {TESTSETS_DIR}')",
    )
    src = src.replace(
        "完整(DNS 英文训练 + WenetSpeech 中文评测)",
        "完整(DNS 训练/质量 + AISHELL 受控 CER + WenetSpeech 真实 CER)",
    )
    assertion = "assert not QUICK_TEST, 'V1 不支持 QUICK_TEST；请改用 SMOKE_RUN=True 做小规模闭环'"
    if assertion not in src:
        src = src.replace(
            "assert DATA_MODE in ('hybrid', 'local'), 'DATA_MODE 只能是 hybrid / local'",
            "assert DATA_MODE in ('hybrid', 'local'), 'DATA_MODE 只能是 hybrid / local'\n" + assertion,
        )
    nb["cells"][2]["source"] = _lines(src)


def update_01() -> None:
    nb = _load("01_data_prep.ipynb")
    _set(nb, 0, "markdown", r"""
# 01 · 数据准备（Colab）

下载 DNS5、AISHELL-1 test 与 WenetSpeech meeting，生成职责互不混淆的三套 V1 测试集。

| 集合 | 来源 | 只回答什么问题 |
|---|---|---|
| `dns_objective` | DNS5 留出干净语音 + 留出真实噪声 + RT60 匹配 RIR | 去噪是否改善 SI-SDR/STOI/PESQ |
| `aishell_controlled` | AISHELL-1 干净中文 + 同一噪声/RIR 矩阵 | 受控条件下中文 CER 是否改善 |
| `wenetspeech_real` | WenetSpeech `test_meeting` 原始录音 | 真实会议录音增强前后 CER；不二次加噪 |

V1 的任务边界是**单通道实时去噪**。RIR 用于模拟混响环境与检验鲁棒性，参考目标
是“带混响、无加性噪声”的语音；当前不宣称模型去混响。

真实 RIR 会先按**实测 RT60**分到 0.2/0.4/0.6/0.8 秒四档，再和相同档位的
合成 RIR 比较，避免旧版“合成上限 0.91 秒、真实中位 2.20 秒”的混淆变量。

执行前：上传最新 `rtse-colab.zip`，先保持 `SMOKE_RUN=True` 跑通，再切正式规模。
""")
    _update_common_config(nb)
    _set(nb, 7, "markdown", r"""
## 2. 下载两套中文评测语料

- **AISHELL-1 test**：安静室内、高保真麦克风录制，用于可控中文 CER。这里使用
  Hugging Face 的 test-only parquet 镜像第一片（约 398 MB，约 2400 条），不用下载
  OpenSLR 15 GB 全量包；它足够生成本项目的小规模固定测试集。
- **WenetSpeech test_meeting**：真实会议录音，用于真实场景 CER。保持原始音频，
  不再叠加 DNS 噪声或 RIR。
""")
    _set(nb, 8, "code", r"""
import pandas as pd

AISHELL_URL = ('https://huggingface.co/datasets/TwinkStart/AISHELL-1/resolve/'
               'refs%2Fconvert%2Fparquet/default/test/0000.parquet')
WNS_URL = ('https://huggingface.co/datasets/lmms-lab/WenetSpeech/resolve/main/'
           'data/test_meeting-00000-of-00001.parquet')

def fetch_parquet(name, url):
    path = f'{ARCHIVE_DIR}/{name}.parquet'
    if not os.path.exists(path):
        print(f'[get   ] {name} ← {url}')
        rc = os.system(f'wget --show-progress -c -T 60 -O {shq(path)} {shq(url)}')
        assert rc == 0 and os.path.getsize(path) > 1e6, f'{name} 下载失败；直接重跑可断点续传'
    else:
        print(f'[cached] {name} ({os.path.getsize(path)/1e6:.0f} MB)')
    return path

aishell_parquet = fetch_parquet('aishell1_test_0000', AISHELL_URL)
wns_parquet = fetch_parquet('wenetspeech_test_meeting', WNS_URL)
aishell = pd.read_parquet(aishell_parquet)
wns = pd.read_parquet(wns_parquet)

print(f'\nAISHELL-1 子片: {len(aishell)} 条，列 {list(aishell.columns)}')
print(f'WenetSpeech meeting: {len(wns)} 条，列 {list(wns.columns)}')
""")
    _set(nb, 13, "markdown", r"""
## 4. 生成三套固定测试集

两套受控集使用同一个80格噪声/RIR矩阵，并额外加入1格 clean→clean 无害性测试：

`SNR 5档 × 噪声 2类 × RIR来源 2类 × RT60 4档 = 80格，另加 identity = 81格`

- `SMOKE_RUN=True`：每格1条，每套81条；WenetSpeech按短/中/长各10条。
- 正式：每格5条，每套405条；WenetSpeech按短/中/长各100条。

受控集的 `clean` 字段表示**带混响、无加性噪声目标**；WenetSpeech 真实集没有
`clean` 字段，也不会产生虚假的“clean 上界”。
""")
    _set(nb, 14, "code", r"""
import io
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
from rtse.data.benchmarks import BenchmarkSource

MIN_SEG_SEC, MAX_SEG_SEC = 3.0, 15.0

def decode_embedded(rec):
    # 解码 HF parquet 的嵌入音频，统一成 16 kHz 单声道 float64。
    a = rec['audio']
    data, sr = sf.read(io.BytesIO(a['bytes']), dtype='float64', always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        import soxr
        data = soxr.resample(data, sr, 16000, quality='VHQ')
    return np.asarray(data, dtype=np.float64)

def parquet_sources(frame, id_fields, limit):
    out = []
    for i in range(len(frame)):
        rec = frame.iloc[i]
        try:
            audio = decode_embedded(rec)
        except Exception:
            continue
        text = str(rec.get('text') or rec.get('sentence') or '').strip()
        duration = audio.size / 16000
        if not (MIN_SEG_SEC <= duration <= MAX_SEG_SEC) or len(text.replace(' ', '')) < 5:
            continue
        source_id = next((str(rec.get(k)) for k in id_fields if rec.get(k) is not None), str(i))
        out.append(BenchmarkSource(source_id, audio, text))
        if len(out) >= limit:
            break
    return out

need_controlled = 100 if SMOKE_RUN else 600
need_real = 60 if SMOKE_RUN else 500
aishell_sources = parquet_sources(aishell, ('name', 'utt_id'), need_controlled)
wns_sources = parquet_sources(wns, ('utt_id', 'aid'), need_real)

dns_sources = []
if not QUICK_TEST:
    dns_candidates = manifest['speech']['val'][:]
    random.Random(20260818).shuffle(dns_candidates)
    for path in dns_candidates:
        try:
            audio = read_audio(path)
        except Exception:
            continue
        duration = audio.size / 16000
        if MIN_SEG_SEC <= duration <= MAX_SEG_SEC:
            dns_sources.append(BenchmarkSource(Path(path).stem, audio))
        if len(dns_sources) >= need_controlled:
            break

print(f'DNS 留出源语音 {len(dns_sources)} 条')
print(f'AISHELL 受控源语音 {len(aishell_sources)} 条')
print(f'WenetSpeech 真实源语音 {len(wns_sources)} 条')
assert aishell_sources and wns_sources
if not QUICK_TEST:
    assert dns_sources
""")
    _set(nb, 15, "code", r"""
from rtse.data.benchmarks import (
    RT60_BINS, build_real_rir_buckets,
    generate_controlled_benchmark, generate_real_cer_benchmark,
)

PER_CELL = 1 if SMOKE_RUN else 5
WENET_PER_BUCKET = 10 if SMOKE_RUN else 100

if not QUICK_TEST:
    print('按实测 RT60 给真实 RIR 分桶…')
    real_rir_buckets = build_real_rir_buckets(
        rir_test,
        min_per_bucket=3 if SMOKE_RUN else 20,
        max_scan=3000 if SMOKE_RUN else 20000,
    )
    print('真实 RIR 桶:', {k: len(v) for k, v in real_rir_buckets.items()})
    noise_by_kind = {'stationary': st_test, 'nonstationary': ns_test}

    dns_index = generate_controlled_benchmark(
        dns_sources, noise_by_kind, real_rir_buckets, DNS_QUALITY_DIR,
        dataset_id='dns5_objective_v1', purpose='objective_audio_quality',
        source_dataset='DNS5 held-out read_speech', per_cell=PER_CELL, seed=20260818,
    )
    aishell_index = generate_controlled_benchmark(
        aishell_sources, noise_by_kind, real_rir_buckets, AISHELL_CER_DIR,
        dataset_id='aishell1_controlled_v1', purpose='controlled_chinese_cer',
        source_dataset='AISHELL-1 test', per_cell=PER_CELL, seed=20260819,
    )

wenet_index = generate_real_cer_benchmark(
    wns_sources, WENET_REAL_DIR, per_duration_bucket=WENET_PER_BUCKET,
)

if not QUICK_TEST:
    print(f'DNS 客观质量集: {len(dns_index["records"])} 条 → {DNS_QUALITY_DIR}')
    print(f'AISHELL 受控 CER: {len(aishell_index["records"])} 条 → {AISHELL_CER_DIR}')
print(f'WenetSpeech 真实 CER: {len(wenet_index["records"])} 条 → {WENET_REAL_DIR}')
""")
    _set(nb, 16, "markdown", r"""
### 结构性校验

这里不再用 VAD 猜“音频是否被截断”。源语音只按时长筛选、从不裁剪；验证重点是：

1. 受控集确实有80个噪声/RIR格 + 1个identity格，RT60没有再被折叠；
2. 标称、内存实测和写盘后 SNR 对齐；
3. 真实 RIR 的实测 RT60 落在对应桶内；
4. WenetSpeech 没有 `clean` 字段、没有二次 SNR/RIR 字段。
""")
    _set(nb, 17, "code", r"""
import collections, statistics as st
from rtse.audio.io import read_audio
from rtse.data.synth import speech_active_mask

def measured_disk_snr(root, record):
    target = read_audio(Path(root, record['clean']))
    noisy = read_audio(Path(root, record['noisy']))
    signal = target - target.mean()
    noise = noisy - target
    noise = noise - noise.mean()
    mask = speech_active_mask(signal)
    return 10 * np.log10(np.mean(signal[mask] ** 2) / np.mean(noise ** 2))

def validate_controlled(path, expected_purpose):
    idx = json.loads(Path(path, 'index.json').read_text(encoding='utf-8'))
    recs = idx['records']
    assert idx['purpose'] == expected_purpose
    assert idx['reference_is_clean'] is True
    assert idx['strata'] == ['snr', 'noise_kind', 'rir_kind', 'rt60_bucket']
    assert len(recs) == (5 * 2 * 2 * 4 + 1) * PER_CELL
    cells = collections.Counter(tuple(r[k] for k in idx['strata']) for r in recs)
    assert len(cells) == 81 and set(cells.values()) == {PER_CELL}
    mixed = [r for r in recs if r['noise_kind'] != 'none']
    identity = [r for r in recs if r['noise_kind'] == 'none']
    assert len(identity) == PER_CELL and all(r['noisy'].endswith('_input.wav') for r in identity)
    worst_snr = max(abs(r['snr_measured'] - r['snr']) for r in mixed)
    assert worst_snr < 1.0, f'SNR 最大偏差 {worst_snr:.2f} dB'
    # 分层抽查写盘后的成对 WAV，锁定“输入/目标被独立归一化”这类隐蔽错误。
    stride = max(1, len(mixed) // 40)
    disk_probe = mixed[::stride][:40]
    worst_disk_snr = max(abs(measured_disk_snr(path, r) - r['snr']) for r in disk_probe)
    assert worst_disk_snr < 0.25, f'写盘后 SNR 最大偏差 {worst_disk_snr:.2f} dB'
    ranges = {t: (lo, hi) for t, lo, hi in RT60_BINS}
    for r in recs:
        if r['rir_kind'] == 'real':
            lo, hi = ranges[r['rt60_bucket']]
            assert lo <= r['rt60_measured'] < hi
    print(
        f'✓ {idx["dataset_id"]}: {len(recs)} 条 / 81格，'
        f'内存SNR最大偏差 {worst_snr:.2f} dB，写盘抽查 {worst_disk_snr:.2f} dB'
    )

if not QUICK_TEST:
    validate_controlled(DNS_QUALITY_DIR, 'objective_audio_quality')
    validate_controlled(AISHELL_CER_DIR, 'controlled_chinese_cer')

real = json.loads(Path(WENET_REAL_DIR, 'index.json').read_text(encoding='utf-8'))
assert real['reference_is_clean'] is False and real['cer_upper_is_meaningful'] is False
assert all('clean' not in r and 'snr' not in r and 'rir_kind' not in r for r in real['records'])
counts = collections.Counter(r['duration_bucket'] for r in real['records'])
assert counts == {'short': WENET_PER_BUCKET, 'medium': WENET_PER_BUCKET, 'long': WENET_PER_BUCKET}
print(f'✓ WenetSpeech 原始会议集: {len(real["records"])} 条，时长分桶 {dict(counts)}')
""")
    _set(nb, 18, "markdown", r"""
## 5. 打包三套测试集，下载到本地

解压后应得到 `data/testsets/{dns_objective,aishell_controlled,wenetspeech_real}`。
""")
    _set(nb, 19, "code", r"""
!cd "{DRIVE}" && rm -f testsets_v1.zip && zip -q -r testsets_v1.zip testsets && ls -lh testsets_v1.zip
print()
print('下一步：')
print('  1. 继续跑 02_train.ipynb（SMOKE_RUN=True 先验证闭环）')
print(f'  2. 下载 {DRIVE}/testsets_v1.zip，解压到本地 data/ 下')
print('  3. 本地分别运行：')
print('     uv run rtse-eval data/testsets/dns_objective --skip-cer')
print('     uv run rtse-eval data/testsets/aishell_controlled')
print('     uv run rtse-eval data/testsets/wenetspeech_real --skip-objective')
""")
    _save("01_data_prep.ipynb", nb)


def update_02() -> None:
    nb = _load("02_train.ipynb")
    _set(nb, 0, "markdown", r"""
# 02 · 去噪模型训练（Colab）

训练 `crn-nano` / `crn-lite`。V1 的目标是**单通道实时去噪**，不是去混响：
加 RIR 时，训练目标仍是带混响但无加性噪声的 `wet` 语音。

训练使用 DNS5 干净语音、真实噪声和真实 RIR，并在线随机混音。当前配方显式覆盖
高 SNR 与 clean→clean 恒等样本，避免旧模型对干净输入无条件过抑制。

先用 `SMOKE_RUN=True` 跑 `crn-nano` 3 epoch 验证数据、训练、断点与导出；该结果
只证明链路可运行，不用于报告质量。正式训练再切到 False。

产物保存到 Drive：`last.pt` 用于续训，`best.pt` 用于导出，`history.json` 保存曲线。
""")
    _update_common_config(nb)
    if not any("无害性与有效性闸门" in "".join(c.get("source", [])) for c in nb["cells"]):
        nb["cells"][11:11] = [
            _new_cell("markdown", r"""
## 3. 无害性与有效性闸门

训练loss下降不等于模型可部署。导出前同时检查：

- clean→clean：完全干净输入不应再次被重度处理；
- noisy→target：留出混音的SI-SDR应有正增益。

冒烟版只打印诊断；正式版若干净透传中位SI-SDR低于20 dB或去噪增益不为正，
直接停止，不把失败模型导出成“最终模型”。
"""),
            _new_cell("code", r"""
import numpy as np
import torch
from rtse.audio.io import read_audio
from rtse.data.dataset import stft_torch, istft_torch
from rtse.metrics.intrusive import si_sdr

@torch.inference_mode()
def run_wave(model, wave):
    x = torch.as_tensor(wave, dtype=torch.float32, device=next(model.parameters()).device)[None]
    spec = stft_torch(x)
    out_spec = model(spec)
    return istft_torch(out_spec, length=x.shape[-1])[0].cpu().numpy()

gate_results = {}
for name in MODELS:
    model = build_model(name).to('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(f'{CKPT_DIR}/{name}/best.pt', map_location=next(model.parameters()).device,
                    weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()

    clean_scores = []
    for path in manifest['speech']['val'][:10]:
        wave = read_audio(path)
        if wave.size < 16000:
            continue
        wave = wave[:16000 * 4]
        clean_scores.append(si_sdr(wave, run_wave(model, wave)))

    gains = []
    for i in range(10):
        noisy, target = val_ds[i]
        noisy_np, target_np = noisy.numpy(), target.numpy()
        gains.append(si_sdr(target_np, run_wave(model, noisy_np)) - si_sdr(target_np, noisy_np))

    clean_median = float(np.median(clean_scores))
    gain_median = float(np.median(gains))
    gate_results[name] = {'clean_passthrough_si_sdr': clean_median,
                          'noisy_delta_si_sdr': gain_median}
    print(f'{name}: clean透传 {clean_median:.2f} dB；留出混音 ΔSI-SDR {gain_median:+.2f} dB')

    if not SMOKE_RUN:
        assert clean_median >= 20.0, f'{name} 对干净输入仍然过抑制，禁止导出'
        assert gain_median > 0.0, f'{name} 留出混音没有正增益，禁止导出'

Path(f'{DRIVE}/training_gates.json').write_text(
    json.dumps(gate_results, ensure_ascii=False, indent=1), encoding='utf-8')
"""),
        ]
        # 原来的“训练曲线”编号顺延。
        curve = nb["cells"][13]
        curve["source"] = _lines("".join(curve["source"]).replace("## 3. 训练曲线", "## 4. 训练曲线"))
    _save("02_train.ipynb", nb)


def update_03() -> None:
    nb = _load("03_export_eval.ipynb")
    _set(nb, 0, "markdown", r"""
# 03 · ONNX 流式导出 + DNS 客观质量评测

1. 导出单帧步进 ONNX，并验证 PyTorch整段 / PyTorch流式 / ONNX流式三者一致；
2. 只在 `dns_objective` 上计算 SI-SDR、STOI、ESTOI、PESQ。

AISHELL 与 WenetSpeech 的 CER 在本地 `rtse-eval` 运行；WenetSpeech 没有干净参考，
这里禁止对它计算有参考指标。
""")
    _update_common_config(nb)
    src5 = "".join(nb["cells"][5]["source"])
    if "training_gates.json" not in src5:
        guard = r"""
# 正式导出必须先通过 02 的 clean透传 + 去噪正增益闸门。
gate_path = Path(f'{DRIVE}/training_gates.json')
assert gate_path.exists(), '缺少 training_gates.json；先完整运行 02 的无害性与有效性闸门'
training_gates = json.loads(gate_path.read_text(encoding='utf-8'))
if not SMOKE_RUN:
    for name, gate in training_gates.items():
        assert gate['clean_passthrough_si_sdr'] >= 20.0, f'{name} 干净透传未达标，禁止正式导出'
        assert gate['noisy_delta_si_sdr'] > 0.0, f'{name} 去噪无正增益，禁止正式导出'
print('训练闸门:', training_gates)

"""
        nb["cells"][5]["source"] = _lines(guard + src5)
    _set(nb, 8, "markdown", r"""
## 3. 在 DNS 客观质量集上评测（含 PESQ）

这是唯一负责 SI-SDR/STOI/PESQ 的测试集。按噪声平稳性、RIR来源和匹配后的
RT60 桶汇总；中文测试集不在这里冒充干净参考。
""")
    src9 = "".join(nb["cells"][9]["source"])
    src9 = src9.replace("TESTSET_DIR", "DNS_QUALITY_DIR")
    src9 = src9.replace(
        "'noise_kind': r['noise_kind'], 'rir_kind': r['rir_kind'],\n"
        "                     'rt60_measured': r['rt60_measured'],",
        "'noise_kind': r['noise_kind'], 'rir_kind': r['rir_kind'],\n"
        "                     'rt60_bucket': r['rt60_bucket'],\n"
        "                     'rt60_measured': r['rt60_measured'],",
    )
    nb["cells"][9]["source"] = _lines(src9)
    _set(nb, 10, "code", r"""
# 快速汇总；正式报告用本地 rtse-eval 的逐条 JSON。
import collections, statistics as st
print(f"{'method':<12}{'noise':<15}{'rir':<8}{'RT60':>7}{'SI-SDR':>9}{'STOI':>8}{'PESQ':>8}")
print('-' * 68)
agg = collections.defaultdict(list)
for r in rows:
    agg[(r['method'], r['noise_kind'], r['rir_kind'], r['rt60_bucket'])].append(r)
for key in sorted(agg, key=lambda x: tuple(str(v) for v in x)):
    m, nk, rk, rt = key
    g = agg[key]
    pq = [r['pesq'] for r in g if r['pesq'] is not None]
    print(f"{m:<12}{nk:<15}{rk:<8}{rt:>7.1f}"
          f"{st.mean(r['si_sdr'] for r in g):>9.2f}"
          f"{st.mean(r['stoi'] for r in g):>8.3f}"
          f"{(st.mean(pq) if pq else float('nan')):>8.3f}")
""")
    _set(nb, 11, "markdown", r"""
## 4. 打包回传

最后只需下载一个 `rtse_handoff.zip`。解压到新电脑的仓库根目录后，会直接得到：

- `models/`：本轮ONNX、导出验证信息、DNSMOS；
- `checkpoints/`：本轮模型的best/last/history，可续训；
- `data/testsets/`：三套固定测试集；
- `results/`：Colab客观指标、训练闸门、数据清单；
- `HANDOFF_MANIFEST.json`：每个文件的大小与SHA-256，供本地检查传输完整性。

不会包含几十GB的 `archives/` 原始下载缓存，也不会夹带旧模型。
""")
    _set(nb, 12, "code", r"""
import datetime as dt
import hashlib
import zipfile

handoff = Path(WORK) / 'rtse_handoff'
if handoff.exists():
    shutil.rmtree(handoff)
(handoff / 'results').mkdir(parents=True)
(handoff / 'checkpoints').mkdir(parents=True)
(handoff / 'models').mkdir(parents=True)

required = {
    'models': Path(MODEL_DIR),
    'testsets': Path(TESTSETS_DIR),
    'colab_metrics': Path(DRIVE) / 'colab_metrics.json',
    'training_gates': Path(DRIVE) / 'training_gates.json',
    'data_manifest': Path(DRIVE) / 'manifest.json',
}
missing = [f'{name}: {path}' for name, path in required.items() if not path.exists()]
assert not missing, '回传包缺少必要产物：\n' + '\n'.join(missing)

shutil.copytree(required['testsets'], handoff / 'data' / 'testsets')
shutil.copy2(required['colab_metrics'], handoff / 'results' / 'colab_metrics.json')
shutil.copy2(required['training_gates'], handoff / 'results' / 'training_gates.json')
shutil.copy2(required['data_manifest'], handoff / 'results' / 'manifest.json')

# 只打包本轮配置中实际训练的模型，防止Drive里残留的旧权重/checkpoint混入。
export_info = Path(MODEL_DIR) / 'export_info.json'
dnsmos_dir = Path(MODEL_DIR) / 'dnsmos'
assert export_info.exists(), '缺少ONNX导出验证信息 export_info.json'
assert (dnsmos_dir / 'sig_bak_ovr.onnx').exists(), '缺少DNSMOS模型'
shutil.copy2(export_info, handoff / 'models' / 'export_info.json')
shutil.copytree(dnsmos_dir, handoff / 'models' / 'dnsmos')
for name in MODELS:
    onnx_path = Path(MODEL_DIR) / f'{name}.onnx'
    assert onnx_path.exists(), f'{name} ONNX不存在'
    shutil.copy2(onnx_path, handoff / 'models' / onnx_path.name)
    src = Path(CKPT_DIR) / name
    assert (src / 'best.pt').exists() and (src / 'last.pt').exists(), f'{name} checkpoint不完整'
    shutil.copytree(src, handoff / 'checkpoints' / name)

files = sorted(p for p in handoff.rglob('*') if p.is_file())
handoff_meta = {
    'schema_version': 1,
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'smoke_run': SMOKE_RUN,
    'models': list(MODELS),
    'file_count_without_manifest': len(files),
    'files': [
        {
            'path': p.relative_to(handoff).as_posix(),
            'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in files
    ],
}
(handoff / 'HANDOFF_MANIFEST.json').write_text(
    json.dumps(handoff_meta, ensure_ascii=False, indent=1), encoding='utf-8')

archive = Path(DRIVE) / 'rtse_handoff.zip'
archive.unlink(missing_ok=True)
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(p for p in handoff.rglob('*') if p.is_file()):
        zf.write(path, Path('rtse_handoff') / path.relative_to(handoff))

print(f'✓ 完整回传包: {archive}')
print(f'  文件 {len(files) + 1} 个，压缩后 {archive.stat().st_size / 2**20:.1f} MB')
print('下载这一个ZIP即可；解压后把 rtse_handoff/ 内各目录合并到仓库根目录。')
""")
    _save("03_export_eval.ipynb", nb)


def main() -> None:
    update_01()
    update_02()
    update_03()
    print("Updated notebooks 01/02/03 and cleared old outputs.")


if __name__ == "__main__":
    main()
