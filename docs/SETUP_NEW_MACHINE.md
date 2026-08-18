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
| **V1冒烟 ONNX 模型** | ✅ 在仓库里 | `models/crn-nano.onnx`，约475 KB，可验证推理闭环；3 epoch结果不代表最终质量 |
| **正式 ONNX 模型** | ⏸ 待训练 | Nano通过闸门后再训练/导出Lite与Large |
| Colab notebooks | ✅ 在仓库里 | `notebooks/`（三个，按顺序跑） |
| V1冒烟评测结果 | ✅ 在仓库里 | DNS客观、AISHELL受控CER、WenetSpeech真实会议CER |
| **固定测试集** | ❌ 需要下载 | 几百 MB，太大不入库；notebook 01 的产物 |
| DNSMOS 模型 | ❌ 需要下载 | 微软的权重文件，不转发分发 |
| 训练 checkpoint | ❌ 在 Drive 上 | `.pt` 含优化器状态，几十 MB，续训时才需要 |

### 4.1 测试集（要算指标才需要）

从 Google Drive 下载 notebook 03 生成的 `rtse_handoff.zip`，解压后把其中
`rtse_handoff/` 的内容合并到项目根目录。测试集会直接落到
`data/testsets/{dns_objective,aishell_controlled,wenetspeech_real}`，不需要再分别下载。

两套受控集的 `index.json` 每条记录都带 `rt60_bucket` 和
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
`models/*.onnx` 会自动发现当前回传的神经模型；三种 DSP 方法始终可用。
目前顶层仅放V1冒烟Nano，避免把旧数据训练的Lite误混进新版评测。

```bash
uv run rtse-asr transcribe data/demo/chinese_news.wav
```

首次会自动下载 faster-whisper 模型（约 500 MB）到 `~/.cache/huggingface`。

有测试集之后：

```bash
uv run rtse-eval data/testsets/dns_objective --skip-cer --out results/dns_objective.json
uv run rtse-eval data/testsets/aishell_controlled --out results/aishell_controlled.json
uv run rtse-eval data/testsets/wenetspeech_real --skip-objective --out results/wenetspeech_real.json
```

---

## 6. 继续训练（Colab 侧）

1. 本地重新打包代码：

```bash
uv run python scripts/pack_for_colab.py
```

2. 把 `dist/colab_upload/` 下的文件（`rtse-colab.zip` + 三个 `.ipynb`）平铺
   上传到 Drive 的 `MyDrive/Audio AI/RTSE/`，**覆盖旧的**——代码或 notebook
   改过就必须重传，否则 Colab 跑的还是旧逻辑，这个坑踩过：
   MCRA 修复和 ASR 模块都在本地改好了，Colab 那边却还是旧代码，
   跑出来的 `colab_metrics.json` 是过期数字。
3. 按顺序跑三个 notebook。数据设计与理由见
   [`notebooks/README.md`](../notebooks/README.md)，操作细节见
   [`COLAB_GUIDE.md`](COLAB_GUIDE.md)。

---

## 7. 当前进度与待办

看 [`PROGRESS.md`](PROGRESS.md)（已完成）和 [`ISSUES.md`](ISSUES.md)（问题与修复记录）。
接手时最该知道的三件事：

1. **V1 已拆成三套评测**：DNS客观质量、AISHELL受控中文CER、WenetSpeech真实会议CER。
2. **现有 ONNX 是修复训练分布之前的诊断基线**，可跑通链路，但存在干净输入过抑制；
   正式结果要用新的高SNR+恒等样本配方重训。
3. **先保持 `SMOKE_RUN=True`**，确认80格受控矩阵+1格无害性、真实RIR匹配桶、训练和导出全部通过，
   再切正式规模。具体见 [`COLAB_GUIDE.md`](COLAB_GUIDE.md)。
