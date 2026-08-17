# Colab 侧操作手册

> 你要在 Colab 上做的事：**下载数据、合成测试集、训练模型、导出 ONNX、算 PESQ**。
> 其余（推理运行时、指标评测、RTF/延迟测量、CER 对比、Web 演示）全部在本地。

**数据设计的理由**（为什么选这些数据集、为什么这样分层）在
[`notebooks/README.md`](../notebooks/README.md)，本文档只讲**怎么操作**。

---

## 0. 一次性准备

### ① 本地打包代码

```bash
uv run python scripts/pack_for_colab.py
```

产出 `dist/rtse-colab.zip`（约 130 KB，只含代码与配置，不含数据/模型/venv）。

> ⚠️ **代码改过就必须重新打包上传**，否则 Colab 跑的还是旧逻辑。
> 这个坑踩过：本地修好了 MCRA 的 bug、写好了 ASR 模块，Colab 那边还是旧代码，
> 跑出来的指标是过期数字，白等一轮。

### ② 上传到 Google Drive

把 `rtse-colab.zip` 传到 Drive 的项目目录。notebook 里的默认值是：

```python
DRIVE_ROOT = '/content/drive/MyDrive/Audio AI/RTSE'
```

改成你实际的目录即可 —— **其余所有路径都从它派生**，不用逐个改。
**路径里有空格没问题**，所有 shell 命令都做了引号处理。

### ③ 上传 notebook

把 `notebooks/` 下三个 `.ipynb` 传到 Drive（或在 Colab 里用「上传笔记本」打开）。

### ④ 运行时设置

菜单 **代码执行程序 → 更改运行时类型 → 硬件加速器 = GPU**（T4 即可）。

- ❌ **别选 TPU**：本模型是 GRU + 因果卷积的流式结构，TPU 没有收益还要折腾 XLA。
- ❌ **别选 CPU**：会慢几十倍。
- 模型很小（最大 crn-large 也只有 1.77 M 参数），T4 的显存绰绰有余。
  **真正的约束是 vCPU 数量和磁盘**，不是 GPU。

---

## 1. 目录布局

三个 notebook 都有一个**「配置」cell**，所有路径在那里定义，别处不写死。
跑完那个 cell 会把布局打印出来：

```
MyDrive/Audio AI/RTSE/                ← DRIVE_ROOT（持久，不受会话断线影响）
├── rtse-colab.zip                    你手动上传的代码包
├── archives/                         压缩包缓存（hybrid 模式）
│   ├── datasets_fullband.clean_fullband.read_speech_000_*.tar.bz2
│   ├── datasets_fullband.noise_fullband.audioset_*.tar.bz2
│   ├── datasets_fullband.impulse_responses_000.tar.bz2
│   └── wenetspeech_test_meeting.parquet
├── manifest.json                     数据清单（含噪声平稳性分组结果）
├── testset/                          固定测试集（带噪 + 干净参考 + 中文转写）
│   ├── index.json
│   └── audio/*.wav
├── testset.zip                       打包好供下载
├── checkpoints/<模型名>/              ★ 训练结果
│   ├── last.pt                       每 epoch 覆盖写，用于断点续训
│   ├── best.pt                       验证 SI-SDR 最优，导出用它
│   └── history.json                  训练曲线
├── models/<模型名>.onnx               ★ 导出的流式模型
├── colab_metrics.json                Colab 侧算的指标（含 PESQ）
└── logs/

/content/rtse_work/data/              ← 语料解压目标（本地盘，会话结束消失）
├── dns_speech_000/                   DNS 干净语音（英文）
├── dns_noise_audioset_*/             DNS 真实噪声
├── dns_noise_freesound_*/
└── dns_ir/                           DNS 真实房间冲激响应
```

**训练结果永远在 Drive 上**（`checkpoints/`、`models/`），断线不丢。

### 需要你决定的几个开关

```python
DATA_MODE = 'hybrid'      # 语料怎么放
N_SPEECH_SHARDS = 1       # DNS 语音分片数，1 片 ≈ 21 小时
N_AUDIOSET_SHARDS = 2     # DNS 噪声分片（AudioSet 来源）
N_FREESOUND_SHARDS = 1    # DNS 噪声分片（Freesound 来源）
QUICK_TEST = False        # True = 只下 WenetSpeech（约 520 MB）验证链路
```

**`DATA_MODE`**：

| 取值 | 压缩包 | 解压到 | 新会话开销 | 训练读取 |
|---|---|---|---|---|
| **`'hybrid'`**（默认，推荐） | Drive（保留） | 本地盘 | 只解压，几分钟 | 全速 |
| `'local'` | 本地盘（用完删） | 本地盘 | **重新下载** | 全速 |

**为什么默认 hybrid**：Drive 是 FUSE 挂载，创建文件开销极高（每个文件一次 API 往返）。
压缩包（几个大文件，顺序读写）留在 Drive 一次下载永久有效；
解压和训练读取都在本地盘全速进行。代价只是每个新会话多花几分钟解压。

> 下载与解压是**两个独立的完成标记**：`archives/.<name>.downloaded`（Drive，跨会话保留）
> 和 `<DATA>/.<name>.extracted`（本地盘，会话内有效）。
> 新会话里前者还在、后者没了 → 只解压，不重下。

**`QUICK_TEST`**：设成 `True` 跳过 DNS 大文件，只下 WenetSpeech（约 520 MB），
十几分钟就能把「数据 → 训练 → 导出 → 回传」整条链路跑通一遍。
**第一次强烈建议先这样跑**，确认流程没问题再投入几小时下载和训练。

---

## 2. 执行顺序

三个 notebook **必须按顺序**跑，后一个依赖前一个的产出。

| # | Notebook | 做什么 | 耗时 | 产出 |
|---|---|---|---|---|
| 1 | `01_data_prep.ipynb` | 下载语料、噪声平稳性分类、合成固定测试集 | 1~2 h（主要是下载） | `manifest.json`、`testset/` |
| 2 | `02_train.ipynb` | 训练 2~3 档模型 | **看第一个 epoch 的实测值** | `checkpoints/*/best.pt` |
| 3 | `03_export_eval.ipynb` | 导出流式 ONNX + 算 PESQ | 约 20 min | `models/*.onnx`、`colab_metrics.json` |

每个 notebook 的**前几个 cell 是固定的**：挂载 Drive → 配置 → 安装代码 → 自检。

**自检那一步不要跳** —— 它验证 Colab 侧与本地是同一条信号链路
（STFT 逐帧一致、dBFS 标定精确到 0.000 dB）。这里一旦有偏差，
训练出来的模型拿回本地就会掉点，而且极难定位 —— 两边单独看都"没问题"。

### 跑完 01 之后要看的三个数字

1. **说话人数量**：DNS 文件名用 `stem.split('_')[0]` 切说话人 id。
   如果打印出来的说话人数接近文件总数，说明切分规则不对，
   划分会退化成按文件随机划分，**指标会虚高**，必须先修这个再往下走。
2. **噪声平稳性两组的比例**：门限 9.0 dB 是在合成噪声上标定的，
   真实录音分布更连续。两组比例悬殊（比如 9:1）就该调门限，
   否则某一组样本太少，对照没有统计意义。
3. **合成 RIR 的标称 vs 实测 RT60**：notebook 会打印对照。
   偏差超过 ±25% 说明 `make_rir` 又出问题了（这个 bug 犯过一次，见 I-22）。

---

## 3. Colab 会断线 —— 已经处理好了

12 小时上限、闲置回收（约 90 分钟无交互）、GPU 配额，会话随时会没。所以：

- **checkpoint 每个 epoch 都存，且存到 Drive**（不是本地盘，那个会话结束就没）
- 存的不只是权重，还有**优化器动量、学习率调度状态、随机数状态**。
  只存权重的话，续训会因为动量和学习率重置而出现明显的 loss 反弹。
- 断线后**重跑 `02_train.ipynb` 即可**，它会自动检测 `last.pt` 并从断点继续。
- **语料在 hybrid 模式下也不会白丢**：压缩包在 Drive 上，
  新会话重跑下载 cell 只会解压（几分钟），不会重新下载。

急着看全流程能不能通：把 `QUICK_TEST` 设成 `True`，或者把 `EPOCHS` 改成 5 先跑一轮。

### 训练时长怎么估

**不要照搬任何预估值，看实测。** 训练器把每个 epoch 的 `epoch_seconds`
写进了 `checkpoints/<模型名>/history.json`：

```python
import json
h = json.load(open(f'{CKPT_DIR}/crn-nano/history.json'))
print(h[0]['epoch_seconds'], '秒/epoch  →', h[0]['epoch_seconds']*60/3600, '小时跑 60 epoch')
```

**建议的推进节奏**：

1. `QUICK_TEST=True` + `EPOCHS=3` —— 十几分钟，验证整条链路能通
2. `QUICK_TEST=False` + `EPOCHS=5` —— 跑一轮真数据，确认 loss 在降、量出真实 epoch 时长
3. `EPOCHS=60` —— 正式训练，断了就重跑，会自动续训

**先跑 `crn-nano`**（参数量只有 crn-lite 的 1/5）。它跑完就能拿到一套完整的
端到端指标，把整个流程闭环；大模型再慢慢跑。
**有一个能用的模型，远好过两个都卡在半路。**

---

## 4. 导出后必须看的三个数

`03_export_eval.ipynb` 会打印一组一致性校验结果。**只有 PASS 才能下载模型**：

```
ONNX流式 vs PyTorch整段 : 1.252e-06 (相对 3.187e-07)   ← 最关键
PyTorch流式 vs 整段     : 1.192e-06
状态形状稳定            : True
==> PASS
```

为什么必须以**整段推理**为基准：只比"ONNX 流式 vs PyTorch 流式"是不够的。
如果 PyTorch 的流式实现本身就偷看了未来帧，两边会**一致地错**，
而离线指标依然好看 —— 这是最危险的一类 bug，上线才会暴露。

> 本地已经用未训练的随机权重把整条导出链路验证通过了（相对误差 3e-7）。
> 所以这一步大概率直接过；真挂了，问题多半出在你改动过的模型结构上
> （多半是某处加了非因果的 padding）。

---

## 5. 回传清单

跑完三个 notebook 后，从 Drive 下载（路径相对 `DRIVE_ROOT`）：

| Drive 上 | 放到本地 |
|---|---|
| `models/*.onnx` | `models/` |
| `models/dnsmos/sig_bak_ovr.onnx` | `models/dnsmos/` |
| `colab_metrics.json` | `results/` |
| `testset.zip`（notebook 01 生成，单独下载） | 解压到 `data/`，使 `data/testset/index.json` 存在 |

> `archives/` 里的几十 GB 压缩包**不用下载**，它们只在 Colab 上用。

notebook 03 的最后一个 cell 会把前三样打包成 `colab_outputs.zip`，下载一个就够。

放好后本地执行：

```bash
uv run rtse-doctor
```

「Colab 产物」一项应从告警变绿。然后跑完整评测（含 CER）：

```bash
uv run rtse-eval
```

Web 演示的方法下拉框里会自动出现神经模型（`models/*.onnx` 是自动扫描的）。

---

## 6. 磁盘预算

**Colab 侧**：语料压缩包约 20 GB，解压后 40~60 GB。
Colab Pro 的本地盘有 200+ GB，够用。Drive 侧在 hybrid 模式下要放得下压缩包。

**本地侧**：C: 盘紧张（见 [`ENVIRONMENT.md`](ENVIRONMENT.md)），所以：

- **不要**把 DNS / WenetSpeech 原始数据下载到本地，它们只在 Colab 上用
- 只下载 `testset.zip`（几百 MB）和模型（约 10 MB）
- 空间不够的话把 `data/` 迁到别的盘，然后设 `RTSE_DATA_DIR` 环境变量
