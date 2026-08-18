# Colab 侧操作手册

> 你要在 Colab 上做的事：**下载数据、合成测试集、训练模型、导出 ONNX、算 PESQ**。
> 其余（推理运行时、指标评测、RTF/延迟测量、CER 对比、Web 演示）全部在本地。

**数据设计的理由**（为什么选这些数据集、为什么这样分层）在
[`notebooks/README.md`](../notebooks/README.md)，本文档只讲**怎么操作**。

---

## 0. 一次性准备

### ① 本地打包

```bash
uv run python scripts/pack_for_colab.py
```

产出 `dist/colab_upload/`，里面**只有两个文件**：

| 文件 | 说明 |
|---|---|
| `rtse-colab.zip` | 代码包（约 220 KB，只含代码与配置，不含数据/模型/venv） |
| `rtse_colab.ipynb` | **三册合并后的单一 notebook** |

同时也打成一个 `dist/colab_upload.zip`，方便一次性传输。

> **为什么合并成一个 notebook**：Colab 每个会话都会清空 `/content`，
> 开一个 notebook 就要重跑一遍「解压代码包 + 解压 20 GB 语料」，一次约 20 分钟。
> 原来三册分开，跑完整条链路要付三次这个开销。合并后只付一次。
> 三个阶段用 `═══` 分割线隔开，各自的说明写在段首。
>
> `notebooks/` 下的三个分册仍然保留作为编辑源，但**不再上传**。
> 改完分册后跑 `uv run python scripts/build_merged_notebook.py` 重新生成合并版。

> ⚠️ **代码或 notebook 改过就必须重新打包上传**，否则 Colab 跑的还是旧逻辑。
> 这个坑踩过：本地修好了 MCRA 的 bug、写好了 ASR 模块，Colab 那边还是旧代码，
> 跑出来的指标是过期数字，白等一轮。

### ② 上传到 Google Drive

把这两个文件传到 Drive 的 **`colab_upload/`** 子目录下（没有就新建）：

```
MyDrive/Audio AI/RTSE/
├── colab_upload/     ← 传这里：rtse-colab.zip + rtse_colab.ipynb
├── colab_outputs/    ← Colab 自动创建，所有产物都在这，下载就下这一个目录
└── archives/         ← 已下好的约 20 GB 语料缓存，**位置不动**
```

> `archives/` 刻意留在原处：它的下载完成标记是按路径记的，搬走等于标记全部失效，
> 20 GB 要重下一遍。

notebook 里的默认根目录是：

```python
DRIVE_ROOT = '/content/drive/MyDrive/Audio AI/RTSE'
```

改成你实际的目录即可 —— **其余所有路径都从它派生**，不用逐个改。
**路径里有空格没问题**，所有 shell 命令都做了引号处理。

### ③ 打开 notebook

在 Drive 里找到 `colab_upload/rtse_colab.ipynb`，右键「打开方式 → Google Colaboratory」。

### ④ 运行时设置

菜单 **代码执行程序 → 更改运行时类型 → 硬件加速器 = GPU**（T4 即可）。

- ❌ **别选 TPU**：本模型是 GRU + 因果卷积的流式结构，TPU 没有收益还要折腾 XLA。
- ❌ **别选 CPU**：会慢几十倍。
- V1 最大的 crn-lite 约0.57M参数，T4显存绰绰有余。
  **真正的约束是 vCPU 数量和磁盘**，不是 GPU。

---

## 1. 目录布局

notebook 开头的**「配置」cell** 定义了所有路径，别处不写死。
跑完那个 cell 会把布局打印出来：

```
MyDrive/Audio AI/RTSE/                ← DRIVE_ROOT（持久，不受会话断线影响）
├── colab_upload/                     你手动上传的
│   ├── rtse-colab.zip                代码包
│   └── rtse_colab.ipynb              合并后的单一 notebook
├── archives/                         压缩包缓存（hybrid 模式）★位置不动
│   ├── datasets_fullband.clean_fullband.read_speech_000_*.tar.bz2
│   ├── datasets_fullband.noise_fullband.audioset_*.tar.bz2
│   ├── datasets_fullband.impulse_responses_000.tar.bz2
│   ├── aishell1_test_0000.parquet
│   └── wenetspeech_test_meeting.parquet
└── colab_outputs/                    ★ 所有产物，要下载就整个下这一个目录
    ├── manifest.json                 数据清单（含噪声平稳性分组结果）
    ├── testsets/
    │   ├── dns_objective/            客观质量，有无噪参考
    │   ├── aishell_controlled/       受控中文 CER，有无噪参考
    │   └── wenetspeech_real/         原始会议 CER，无 clean 音频上界
    ├── testsets_v1.zip               三套测试集打包
    ├── checkpoints/<模型名>/          训练结果
    │   ├── last.pt                   每 epoch 覆盖写，用于断点续训
    │   ├── best.pt                   验证 SI-SDR 最优，导出用它
    │   └── history.json              训练曲线
    ├── models/<模型名>.onnx           导出的流式模型
    ├── training_gates.json           无害性与有效性闸门结果
    ├── colab_metrics.json            Colab 侧算的指标（含 PESQ）
    ├── rtse_handoff.zip              最终回传包（下载这一个就够）
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
QUICK_TEST = False        # V1 必须保持 False
SMOKE_RUN = True          # 先跑小规模闭环；正式结果再改 False
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

**`SMOKE_RUN`**：V1 的推荐起点。下载仍使用同一批 DNS 最小分片，但生成量降为
每套受控集81条（含clean→clean无害性格）、真实会议30条，训练只跑Nano 3 epoch。
它用于验证链路，结果不作数。
`QUICK_TEST` 已停用，因为没有 DNS 留出噪声/RIR 就无法验证受控评测和训练闭环。

---

## 2. 执行顺序

先跑**初始化**的 4 个 cell，然后三个 PART **必须按顺序**跑，后一段依赖前一段的产出。

| 段 | 做什么 | 耗时 | 产出（都在 `colab_outputs/`） |
|---|---|---|---|
| 初始化 | 挂载 Drive → 配置 → 安装代码 → 自检 | 约 20 min（主要是解压） | — |
| **PART 1** 数据准备 | 下载语料、分类噪声、生成三套固定测试集 | 1~2 h（主要是下载） | `manifest.json`、`testsets_v1.zip` |
| **PART 2** 训练 | 训练模型 + 过闸门 | **看第一个 epoch 的实测值** | `checkpoints/*/best.pt`、`training_gates.json` |
| **PART 3** 导出评测 | 导出流式 ONNX + 在 DNS 客观集算 PESQ + 打包 | 约 20 min | `models/*.onnx`、`colab_metrics.json`、`rtse_handoff.zip` |

初始化那 4 个 cell **整个流程只跑一次**——这正是把三册合并成一个 notebook 的原因：
`/content` 每个会话都会清空，原来三册分开要付三次这 20 分钟。

**自检那一步不要跳** —— 它验证 Colab 侧与本地是同一条信号链路
（STFT 逐帧一致、dBFS 标定精确到 0.000 dB）。这里一旦有偏差，
训练出来的模型拿回本地就会掉点，而且极难定位 —— 两边单独看都"没问题"。

### 跑完 01 之后要看的四项校验

1. **说话人数量**：DNS 文件名必须用 `reader_(\d+)` 提取说话人；notebook 有断言。
2. **噪声平稳性两组的比例**：门限 9.0 dB 是在合成噪声上标定的，
   真实录音分布更连续。两组比例悬殊（比如 9:1）就该调门限，
   否则某一组样本太少，对照没有统计意义。
3. **80个实验格是否齐全**：RT60 必须是独立分层字段，不能再折叠进 `synth`。
4. **真实 RIR 是否落在匹配桶**：0.2/0.4/0.6/0.8秒分别与同档合成RIR比较。

---

## 3. Colab 会断线 —— 已经处理好了

12 小时上限、闲置回收（约 90 分钟无交互）、GPU 配额，会话随时会没。所以：

- **checkpoint 每个 epoch 都存，且存到 Drive**（不是本地盘，那个会话结束就没）
- 存的不只是权重，还有**优化器动量、学习率调度状态、随机数状态**。
  只存权重的话，续训会因为动量和学习率重置而出现明显的 loss 反弹。
- 断线后**重跑 PART 2 即可**（初始化那 4 个 cell 要先跑），它会自动检测 `last.pt` 并从断点继续。
- **语料在 hybrid 模式下也不会白丢**：压缩包在 Drive 上，
  新会话重跑下载 cell 只会解压（几分钟），不会重新下载。

急着看全流程能不能通：保持 `SMOKE_RUN=True`，它会自动使用 Nano、3 epoch 和小测试集。

### 训练时长怎么估

**不要照搬任何预估值，看实测。** 训练器把每个 epoch 的 `epoch_seconds`
写进了 `checkpoints/<模型名>/history.json`：

```python
import json
h = json.load(open(f'{CKPT_DIR}/crn-nano/history.json'))
print(h[0]['epoch_seconds'], '秒/epoch  →', h[0]['epoch_seconds']*60/3600, '小时跑 60 epoch')
```

**建议的推进节奏**：

1. `SMOKE_RUN=True` —— Nano 3 epoch，验证整条链路
2. `SMOKE_RUN=False`，先把正式 `EPOCHS` 临时设为5 —— 确认新分布下 loss 与无害性
3. 恢复60 epoch —— 正式训练，断了重跑会自动续训

**先跑 `crn-nano`**（参数量只有 crn-lite 的 1/5）。它跑完就能拿到一套完整的
端到端指标，把整个流程闭环；大模型再慢慢跑。
**有一个能用的模型，远好过两个都卡在半路。**

---

## 4. 导出后必须看的三个数

PART 3 会打印一组一致性校验结果。**只有 PASS 才能下载模型**：

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

跑完三个 PART 后，只下载 `colab_outputs/` 下的 **`rtse_handoff.zip`**。PART 3
最后一个 cell 会先核对必要产物，再生成如下结构：

```text
rtse_handoff/
├── HANDOFF_MANIFEST.json       文件大小与SHA-256
├── models/                     ONNX、导出验证、DNSMOS
├── checkpoints/<本轮模型>/     best.pt、last.pt、history.json
├── data/testsets/              DNS、AISHELL、WenetSpeech三套固定测试集
└── results/
    ├── colab_metrics.json
    ├── training_gates.json
    └── manifest.json
```

把 `rtse_handoff/` 内的目录合并到本地仓库根目录即可。包里不会包含 `archives/`
几十GB的原始下载缓存，也只收集本轮 `MODELS` 声明的checkpoint，避免旧模型混入。

放好后本地执行：

```bash
uv run rtse-doctor
```

然后分三次跑评测，输出文件不要共用：

```bash
uv run rtse-eval data/testsets/dns_objective --skip-cer --out results/dns_objective.json
uv run rtse-eval data/testsets/aishell_controlled --out results/aishell_controlled.json
uv run rtse-eval data/testsets/wenetspeech_real --skip-objective --out results/wenetspeech_real.json
```

正式报告再用更强的faster-whisper `medium`复核前三名方法，并写到**不同输出文件**：

```bash
uv run rtse-eval data/testsets/aishell_controlled --methods none,specsub,crn-nano --asr-model medium --out results/aishell_medium.json
uv run rtse-eval data/testsets/wenetspeech_real --methods none,specsub,crn-nano --asr-model medium --skip-objective --out results/wenet_medium.json
```

评测器会把ASR规格与每格抽样数写进缓存；配置不一致时拒绝续跑，避免把small和medium混成一张表。

Web 演示的方法下拉框里会自动出现神经模型（`models/*.onnx` 是自动扫描的）。

---

## 6. 磁盘预算

**Colab 侧**：DNS压缩包约20 GB，另加两份中文 parquet 约0.62 GB；解压后约40~60 GB。
Colab Pro 的本地盘有 200+ GB，够用。Drive 侧在 hybrid 模式下要放得下压缩包。

**本地侧**：C: 盘紧张（见 [`ENVIRONMENT.md`](ENVIRONMENT.md)），所以：

- **不要**把 DNS / AISHELL / WenetSpeech 原始数据下载到本地，它们只在 Colab 上用
- 只下载 `testsets_v1.zip` 和模型（约10 MB）
- 空间不够的话把 `data/` 迁到别的盘，然后设 `RTSE_DATA_DIR` 环境变量
