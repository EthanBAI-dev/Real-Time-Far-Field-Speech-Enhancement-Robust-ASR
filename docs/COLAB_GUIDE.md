# Colab 侧操作手册

> 你要在 Colab 上做的事：**下载数据、合成数据集、训练模型、导出 ONNX、算 PESQ**。
> 其余（推理运行时、指标评测、RTF/延迟测量、ASR 对比、Web 演示）全部在本地，已经做完并验证。

---

## 0. 一次性准备

### ① 在本地打包代码

```bash
uv run python scripts/pack_for_colab.py
```

产出 `dist/rtse-colab.zip`（约 127 KB，只含代码与配置，不含数据/模型/venv）。

### ② 上传到 Google Drive

把 `rtse-colab.zip` 传到 Drive 的项目目录。notebook 里的默认值是：

```python
DRIVE_ROOT = '/content/drive/MyDrive/Audio AI/RTSE'
```

改成你实际的目录即可 —— **其余所有路径都从它派生**，不用逐个改。
**路径里有空格没问题**，所有 shell 命令都做了引号处理（已用带空格的路径实测验证）。

> 配置 cell 会先断言这个目录存在。写错时会直接报错并列出该目录下现有的文件，
> 而不是等到后面某一步才莫名其妙地失败。

### ③ 上传 notebook

把 `notebooks/` 下三个 `.ipynb` 传到 Drive（或直接在 Colab 里用「上传笔记本」打开）。

### ④ Colab 运行时设置

菜单 **代码执行程序 → 更改运行时类型 → 硬件加速器 = GPU**（T4 即可）。

---

## 1. 目录布局

三个 notebook 都有一个**「配置」cell**，所有路径都在那里定义，别处不写死路径。
跑完那个 cell 会把布局打印出来：

```
MyDrive/Audio AI/RTSE/                ← DRIVE_ROOT（持久，不受会话断线影响）
├── rtse-colab.zip                    你手动上传的代码包
├── archives/                         语料压缩包缓存（hybrid 模式，约 23 GB）
│   ├── data_thchs30.tgz
│   ├── musan.tar.gz
│   └── rirs_noises.zip
├── manifest.json                     数据清单（按说话人划分好的文件列表）
├── testset/                          固定测试集（带噪 + 干净参考 + 转写）
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
├── data_thchs30/                     中文语音 + 转写
├── musan/noise/                      环境噪声
├── RIRS_NOISES/                      房间冲激响应
└── synth_impulsive/                  本项目合成的冲激噪声
```

**训练结果永远在 Drive 上**（`checkpoints/`、`models/`），断线不丢。

### 两个需要你决定的开关

```python
DATA_MODE  = 'hybrid'   # 语料怎么放
QUICK_TEST = False      # 是否只跑小语料验证流程
```

**`DATA_MODE`** —— 约 23 GB 的原始语料怎么放：

| 取值 | 压缩包 | 解压到 | 新会话开销 | 训练读取速度 |
|---|---|---|---|---|
| **`'hybrid'`**（默认，推荐） | Drive（保留） | 本地盘 | 只解压，几分钟 | 全速 |
| `'local'` | 本地盘（用完删） | 本地盘 | **重新下载 15~40 分钟** | 全速 |
| `'drive'` | Drive（保留） | **Drive** | 无 | 慢 2~5 倍 |

**为什么默认 hybrid 而不是 drive**（即使你的 Drive 空间充足）：

Drive 是 FUSE 挂载，**创建文件的开销极高** —— 每个文件都是一次 API 往返。
THCHS-30 有一万多个小 wav，直接解压到 Drive 可能要**几个小时**，
而解压到本地盘只要几分钟。训练时按文件随机读取同样会被 FUSE 拖慢 2~5 倍。

hybrid 把两者的优点都拿到了：压缩包（3 个大文件，顺序读写，FUSE 很擅长）留在 Drive
一次下载永久有效；解压和训练读取都在本地盘全速进行。
代价只是每个新会话多花几分钟解压。

> 下载与解压是**两个独立的完成标记**：
> `archives/.<name>.downloaded`（在 Drive，跨会话保留）和
> `<DATA>/.<name>.extracted`（本地盘，会话内有效）。
> 新会话里前者还在、后者没了 → 只解压，不重下。

> `02_train.ipynb` 会在建数据集**之前**先检查语料文件是否还在，
> 不在就直接报错并告诉你当前模式下要花多久恢复 —— 不会等训练跑一半才炸。

**`QUICK_TEST`** —— 设成 `True` 只下载 337 MB 的 LibriSpeech dev-clean，
约 15 分钟就能把「数据 → 训练 → 导出 → 回传」整条链路跑通一遍。
**第一次强烈建议先这样跑一遍**，确认流程没问题再投入几小时下载和训练。
它是英文语料，算不了中文 CER，只用来验证流程。

---

## 2. 执行顺序

三个 notebook **必须按顺序**跑，后一个依赖前一个的产出。

| # | Notebook | 做什么 | 耗时 | 产出（都在 Drive） |
|---|---|---|---|---|
| 1 | `01_data_prep.ipynb` | 下载语料、按说话人划分、合成固定测试集 | 1~1.5 h（主要是下载）<br>QUICK_TEST 约 15 min | `manifest.json`、`testset/` |
| 2 | `02_train.ipynb` | 训练 2~3 档模型 | **看第一个 epoch 的实测值**（见第 9 节） | `checkpoints/*/best.pt` |
| 3 | `03_export_eval.ipynb` | 导出流式 ONNX + 算 PESQ | 约 20 min | `models/*.onnx`、`colab_metrics.json` |

每个 notebook 的**前几个 cell 是固定的**：
挂载 Drive → **配置** → 安装代码 → **自检**。

自检那一步不要跳，它验证 Colab 侧与本地是同一条信号链路
（STFT 逐帧一致、dBFS 标定精确到 0.000 dB）。
这里一旦有偏差，训练出来的模型拿回本地就会掉点，而且极难定位 ——
两边单独看都"没问题"。

### 下载的健壮性

下载部分做了三件事，断线重来不会从头开始：

- **断点续传**（`wget -c`）
- **镜像回退**：openSLR 主站在亚洲经常很慢，失败会自动依次试
  `openslr.elda.org`（欧洲）和 `openslr.magicdatatech.com`（中国）
- **`.done` 标记 + 解压后校验**：已完成的跳过；压缩包被截断时会立刻报错，
  而不是等到后面以"扫到 0 个文件"的形式暴露

下载前会先检查磁盘空间（按解压峰值算，需要约 2 倍体积），不够就直接提示换方案。

---

## 3. 数据集选型

| 用途 | 数据集 | 体积 | 为什么选它 |
|---|---|---|---|
| 中文语音 | **THCHS-30**（openSLR SLR18） | ~6.4 GB | 带完整转写，可直接算 CER；比 AISHELL-1（15 GB）小得多 |
| 噪声 | **MUSAN** 的 noise 子集（SLR17） | ~11 GB | 真实录制的环境噪声，种类全 |
| 房间冲激响应 | **RIRS_NOISES**（SLR28） | ~6 GB | 含真实测量 + 仿真 RIR，"远场"必需 |
| 冲激型噪声 | **本项目合成** | 0 | 见下 |

**MUSAN 只取 noise 子集**，不要 music 和 speech：music 会让模型学会压制音乐（本项目不需要），
speech 子集会与目标语音混淆。

### 为什么一定要额外合成冲激型噪声

本地实验已经跑出结论（[`FINDINGS.md`](FINDINGS.md) F-02）：
键盘敲击这类冲激噪声会让**所有** MCRA 类 DSP 方法失效 —— ΔSI-SDR 为负，
谱减法甚至掉 0.7 dB。根因是 MCRA 假设"噪声是谱的下包络、语音是其上的瞬态"，
而冲激噪声在它看来和语音起始一模一样，于是噪声估计**根本不更新**。

**这正是神经网络最有说服力的立足点** —— 数据驱动的模型不依赖这个假设。
但前提是训练集里得有这类噪声。MUSAN 里这类样本偏少，所以 notebook 01 会额外合成
7 类 × 30 条并入训练/测试。这一步不能省，否则模型也学不会，
"NN 打败 DSP"这个最有力的论点就没了。

---

## 4. 几个关键设计，别改错

### 数据划分按**说话人**，不是按文件

按文件随机划分会让同一个说话人同时出现在训练集和测试集里，
模型能靠"记住这个人的音色"作弊，指标虚高。noise 和 RIR 同理，测试用的必须是训练没见过的。

### 训练集**在线随机混音**，测试集**固定预生成**

- 训练：每个 epoch 见到的组合都不同，等价于把数据量放大了
  （语音数 × 噪声数 × SNR 档数 × RIR 数）倍 —— 对小规模子集尤其关键。
- 测试：必须固定存盘。否则每次评测的噪声段和 SNR 都不同，前后两次的 CER 没有可比性。

### 参考信号 = **混响后**的干净语音

加了混响时，参考不是原始干信号。否则降噪模型会因为"没能去掉混响"被扣分，
把降噪和去混响两件事混在一起 —— 而我们并没有为去混响准备足够的数据。
（想训去混响就把 `dereverb_target=True` 打开。）

### SNR 只在**语音活跃段**上定义

DNS Challenge 的约定。把静音段也算进语音功率的话，同一个"0 dB"在"话密"和"话稀"
两段录音上实际信噪比能差 5 dB 以上，整个 SNR 维度就没有可比性了。
本地和 Colab 用的是**同一个** `mix_at_snr` 函数，保证两边的"0 dB"是同一件事。

---

## 5. Colab 会断线 —— 已经处理好了

12 小时上限、闲置回收、GPU 配额，会话随时会没。所以：

- **checkpoint 每个 epoch 都存，且存到 Drive**（不是 Colab 本地盘，那个会话结束就没）
- 存的不只是权重，还有**优化器动量、学习率调度状态、随机数状态**。
  只存权重的话，续训会因为动量和学习率重置而出现明显的 loss 反弹。
- 断线后**重跑 `02_train.ipynb` 即可**，它会自动检测 `last.pt` 并从断点继续。

**语料在默认的 hybrid 模式下也不会白丢**：压缩包在 Drive 上，
新会话重跑 01 的「下载语料」cell 只会解压（几分钟），不会重新下载。

> `02_train.ipynb` 会在建数据集之前**先检查语料文件是否还在**，
> 不在就直接报错并告诉你当前模式下要花多久恢复 —— 不会等训练跑一半才炸。

急着看全流程能不能通的话，两个办法：
把 `QUICK_TEST` 设成 `True`（小语料，15 分钟走完全程），
或者把 `EPOCHS` 改成 15~20 先跑一轮。

---

## 6. 导出后必须看的三个数

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

> 本地已经用未训练的随机权重模型把整条导出链路验证通过了
> （相对误差 3e-7，PASS）。所以这一步大概率会直接过；真挂了，问题多半出在
> 你改动过的模型结构上（多半是某处加了非因果的 padding）。

---

## 7. 回传清单

跑完三个 notebook 后，从 Drive 下载：

下面的路径都相对于 `DRIVE_ROOT`（默认 `MyDrive/Audio AI/RTSE`）。
notebook 03 的最后一个 cell 会把前三样打包成 `colab_outputs.zip`，下载一个就够。

| Drive 上 | 放到本地 |
|---|---|
| `models/*.onnx` | `models/` |
| `models/dnsmos/sig_bak_ovr.onnx` | `models/dnsmos/` |
| `colab_metrics.json` | `results/` |
| `testset.zip`（notebook 01 生成，单独下载） | 解压到 `data/`，使 `data/testset/index.json` 存在 |

> `archives/` 里的 23 GB 压缩包**不用下载**，它们只在 Colab 上用。

放好后本地执行：

```bash
uv run rtse-doctor
```

「Colab 产物」一项应从告警变绿；Web 演示的方法下拉框里会自动出现神经模型
（`models/*.onnx` 是自动扫描的，不用改任何代码）。

---

## 8. 磁盘预算提醒

本地 C: 盘只剩约 16 GB（见 [`ENVIRONMENT.md`](ENVIRONMENT.md)），所以：

- **测试集控制在 2 GB 以内**（notebook 01 里 `PER_CELL=20`、`SEG_SEC=6.0` 就是按这个定的，
  会打印预估体积，跑之前看一眼）
- **不要**把 THCHS-30 / MUSAN / RIRS 原始数据下载到本地，它们只在 Colab 上用
- 模型很小（crn-lite 的 ONNX 约 2.2 MB），随便传

空间实在不够的话，把项目的 `data/` 迁到 D: 或 S:，然后设环境变量：

```bash
setx RTSE_DATA_DIR "D:\rtse_data"
```

---

## 9. Colab 免费版（T4 + 2 vCPU）实操建议

### 运行时怎么选

**硬件加速器选 T4 就对了**，免费层没有更好的选项（A100 / L4 属于 Pro / Pro+）。

- ❌ **别选 TPU**：本模型是 GRU + 因果卷积的流式结构，TPU 没有收益，
  还要额外折腾 PyTorch/XLA。
- ❌ **别选 CPU**：会慢几十倍。
- 「高 RAM」选项是 Pro 专属，免费版看不到，也不需要 —— 本项目模型很小
  （最大的 crn-large 也只有 1.77 M 参数），T4 的 16 GB 显存绰绰有余。

**真正的约束不是 GPU，是这几条**：

| 约束 | 免费版实际情况 | 影响 |
|---|---|---|
| vCPU | **2 个** | 决定数据加载能不能喂饱 GPU |
| 系统内存 | 约 12.7 GB | 够用 |
| 本地磁盘 | 约 110 GB | 够放 23 GB 语料（你实测 66 GB 可用） |
| 闲置回收 | 约 90 分钟无交互 | 挂机会被收走，checkpoint 已应对 |
| 会话上限 | 名义 12 h，实际常更短 | 断点续训已应对 |
| 动态用量上限 | 连续重度使用会被限流一段时间 | 别无谓地反复重跑 |

### 数据加载会不会拖后腿

**不会。** 已实测（i7-14700，`reverb_prob=0.5`）：

| 指标 | 数值 |
|---|---|
| 单样本合成 | 5.1 ms |
| 2 worker 吞吐 | 393 样本/秒 |
| 20000 样本 epoch 的加载耗时 | **0.8 分钟** |

即便 Colab 的 CPU 慢 3 倍，也只是 2~3 分钟/epoch，不会成为瓶颈。
所以 **`num_workers=2` 就够**，调大反而会在 2 个 vCPU 上互相抢占。

> 这个结论来之不易 —— 之所以去测，是因为「免费版只有 2 vCPU」这个问题。
> 一测就发现卷混响用的是直接卷积，T60=0.6 s 时**单个样本要 7 秒**，
> 一个 epoch 光加载就要 5 小时、GPU 全程空转。已改用 FFT 卷积（快 3500 倍）
> 并加了性能回归测试。详见 [`ISSUES.md`](ISSUES.md) I-19。
>
> 这类问题最阴险的地方是**不报错**，只表现为"训练慢得离谱"，
> 而第一反应往往是怀疑 GPU 不够快、去换更贵的运行时 —— 方向完全错了。

### 训练时长怎么估

**不要照搬任何预估值，看实测。** 训练器把每个 epoch 的 `epoch_seconds`
写进了 `checkpoints/<模型名>/history.json`：

```python
import json
h = json.load(open(f'{CKPT_DIR}/crn-lite/history.json'))
print(h[0]['epoch_seconds'], '秒/epoch  →', h[0]['epoch_seconds']*60/3600, '小时跑 60 epoch')
```

第一个 epoch 跑完就能算出总时长，据此决定 `EPOCHS` 定多少、要分几次会话。

**建议的推进节奏**：

1. `QUICK_TEST=True` + `EPOCHS=3` —— 15 分钟，验证整条链路能通
2. `QUICK_TEST=False` + `EPOCHS=5` —— 跑一轮真数据，确认 loss 在降、量出真实的 epoch 时长
3. `EPOCHS=60` —— 正式训练，断了就重跑，会自动续训

**先跑 `crn-nano`**（参数量只有 crn-lite 的 1/5）。它跑完就能拿到一套完整的
端到端指标，把整个流程闭环；`crn-lite` 再慢慢跑。
有一个能用的模型，远好过两个都卡在半路。
