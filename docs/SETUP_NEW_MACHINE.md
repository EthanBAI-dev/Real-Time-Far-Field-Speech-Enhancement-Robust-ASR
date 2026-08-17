# 换一台机器继续干活

从零把这个项目在新机器上跑起来。**已实测**：全新 clone → 装依赖 → 跑测试 → 出增强音频，
不需要任何来自旧机器的手工拷贝。

---

## 1. 前置：只需要 uv

```bash
winget install astral-sh.uv
```

不需要系统 Python、不需要 conda、不需要 MSVC——`uv` 会自己拉一个独立的 Python 3.11
（版本锁死的原因见 [`ENVIRONMENT.md`](ENVIRONMENT.md)：3.12+ 在 Windows 上
torch / onnxruntime / pesq 缺 wheel，一旦回落到源码编译必然失败）。

macOS / Linux 用 `curl -LsSf https://astral.sh/uv/install.sh | sh`。

## 2. Clone + 装依赖

```bash
git clone https://github.com/EthanBAI-dev/Real-Time-Far-Field-Speech-Enhancement-Robust-ASR.git
```

```bash
cd Real-Time-Far-Field-Speech-Enhancement-Robust-ASR && uv sync --extra dev
```

首次会下载约 500 MB 依赖（torch CPU 版已锁定，不会拉 2.5 GB 的 CUDA 变体）。

## 3. 验证

```bash
uv run pytest -q
```

应当全绿（约 160 项，含 2 项跳过的网络门控测试 + 1 项 xfail）。**跑通到这里，
说明代码链路完整**——模型定义、DSP、STFT、ONNX 导出、ASR 归一化全部可用。

```bash
uv run rtse-doctor
```

会列出环境自检结果。此时「Colab 产物」一项是**黄的**，因为测试集还没下载（见下一节）。

---

## 4. 仓库里已经带了什么 / 还缺什么

| | 状态 | 说明 |
|---|---|---|
| 全部源码 | ✅ 在仓库里 | |
| **训练好的 ONNX 模型** | ✅ 在仓库里 | `models/crn-{nano,lite,large}.onnx`，约 9.5 MB，**clone 下来就能直接推理** |
| Colab notebooks | ✅ 在仓库里 | `notebooks/`（三个，按顺序跑） |
| 评测结果 json | ⏸ 已清空 | 旧数据集的结果已移除，等新数据跑出 |
| **固定测试集** | ❌ 需要下载 | 几百 MB，太大不入库；notebook 01 的产物 |
| DNSMOS 模型 | ❌ 需要下载 | 微软的权重文件，不转发分发 |
| 训练 checkpoint | ❌ 在 Drive 上 | `.pt` 含优化器状态，几十 MB，续训时才需要 |

### 4.1 测试集（要算指标才需要）

从 Google Drive 下载 `testset.zip`（`MyDrive/Audio AI/RTSE/` 下，是 notebook 01 的产物），
解压到项目根的 `data/`，使 `data/testset/index.json` 存在。

拿到后先看一眼 `index.json`：每条记录都带 `rt60_nominal`（标称）和
`rt60_measured`（Schroeder 反向积分实测）。**两者应当接近**——踩过一次坑
（I-22：标称 0.9 s 实测只有 0.63 s），此后一律以实测为准。

### 4.2 DNSMOS（要算无参考 MOS 才需要）

```bash
mkdir -p models/dnsmos && curl -L -o models/dnsmos/sig_bak_ovr.onnx https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
```

拿不到也不影响主流程，评测报告里该列会自动标 n/a。

### 4.3 训练 checkpoint（只有续训才需要）

在 Drive 的 `MyDrive/Audio AI/RTSE/checkpoints/<模型名>/`。
**只是想继续训练的话不用下载到本地**——直接在 Colab 上跑 `02_train.ipynb`，
它会自己从 Drive 读 `last.pt` 续训。

---

## 5. 马上能做的事

```bash
uv run rtse-enhance --method wiener data/demo/chinese_news.wav out.wav
```

```bash
uv run rtse-server
```

启动 Web 演示（浏览器打开 http://127.0.0.1:8000 ），方法下拉框会自动扫描
`models/*.onnx`，三档神经模型和三种 DSP 方法都在。

```bash
uv run rtse-asr transcribe data/demo/chinese_news.wav
```

首次会自动下载 faster-whisper 模型（约 500 MB）到 `~/.cache/huggingface`。

有测试集之后：

```bash
uv run rtse-eval --cer-per-cell 2
```

---

## 6. 继续训练（Colab 侧）

1. 本地重新打包代码：

```bash
uv run python scripts/pack_for_colab.py
```

2. 把 `dist/rtse-colab.zip` 上传到 Drive 的 `MyDrive/Audio AI/RTSE/`，**覆盖旧的**。
   代码改过就必须重传，否则 Colab 跑的还是旧逻辑——这个坑踩过：
   MCRA 修复和 ASR 模块都在本地改好了，Colab 那边却还是旧代码，
   跑出来的 `colab_metrics.json` 是过期数字。
3. 按顺序跑 `notebooks/` 下的三个 notebook。数据设计与理由见
   [`notebooks/README.md`](../notebooks/README.md)，操作细节见
   [`COLAB_GUIDE.md`](COLAB_GUIDE.md)。

---

## 7. 当前进度与待办

看 [`PROGRESS.md`](PROGRESS.md)（已完成）和 [`ISSUES.md`](ISSUES.md)（问题与修复记录）。
接手时最该知道的三件事：

1. **数据集刚换过（2026-08-07），新流水线还没在 Colab 上实跑过。**
   现在是 DNS Challenge 英文训练 + WenetSpeech 中文评测（跨语种泛化论证）。
   之前基于 THCHS-30 + MUSAN 的两版已废弃删除，连带**所有旧指标和实验发现都已清空**——
   `METRICS.md` 的主表和 `FINDINGS.md` 的 F-01~F-08 都是待重测状态。
   旧内容在 git 历史里（`git show ad69d0b:docs/FINDINGS.md`）。
2. **仓库里的 ONNX 模型是旧数据训出来的**，可以用来跑通链路和 Web 演示，
   但**不是新数据集的结果**，正式指标要等重新训练。
3. **跑完 01 要盯三个数字**（说话人数量是否合理、噪声平稳性两组比例、
   合成 RIR 标称 vs 实测 RT60）——具体见 [`COLAB_GUIDE.md`](COLAB_GUIDE.md) 第 2 节。
