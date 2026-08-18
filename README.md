# RTSE

**Real-Time Single-Channel Speech Denoising for Robust Chinese ASR**
单通道实时语音降噪与中文 ASR 鲁棒性系统

[![tests](https://github.com/EthanBAI-dev/Real-Time-Far-Field-Speech-Enhancement-Robust-ASR/actions/workflows/tests.yml/badge.svg)](https://github.com/EthanBAI-dev/Real-Time-Far-Field-Speech-Enhancement-Robust-ASR/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

```
麦克风 → VAD（分段/显示）→ 去噪（DSP vs CRN）→ 中文 ASR → 实时字幕
```

一条**可实时运行、可量化、可演示**的单通道去噪前端链路，用职责分离的指标矩阵回答：
**语音增强让 ASR 字错率降低了多少，代价是多少毫秒延迟和多少 CPU。**

> V1 不宣称去混响。RIR 是训练/测试中的混响环境变量，模型目标是
> “带混响、无加性噪声”的语音。真正的去混响作为后续独立课题，避免把两个任务混在一起。

---

## 快速开始

> 换一台新机器从零开始，看 [docs/SETUP_NEW_MACHINE.md](docs/SETUP_NEW_MACHINE.md)
> ——已实测：clone → 装依赖 → 跑测试 → 出增强音频，全程不需要旧机器上的任何文件。
> 仓库当前随附 **V1 冒烟版 crn-nano ONNX**（约475 KB），clone 后可直接验证
> 流式推理闭环；它只训练3 epoch且未通过质量门槛，不能当作最终模型。
> crn-lite / large 的结构和导出代码已实现，正式权重等待新版配方重训。

```bash
uv sync
```

```bash
uv run rtse-doctor
```

```bash
uv run rtse-server
```

浏览器打开 **http://127.0.0.1:8000** —— 三个页面：实时麦克风演示 `/`、
文件实验台 `/lab`（不需要麦克风）、基准看板 `/bench`。

> 麦克风只在 `localhost` 或 HTTPS 下可用，务必用 `127.0.0.1` 打开而不是局域网 IP。

---

## 当前状态

| 阶段 | 状态 |
|---|---|
| Phase 0 文档与骨架 | ✅ |
| Phase 1 环境搭建 | ✅ `rtse-doctor` 11 项实测检查全通过 |
| Phase 2 信号与 DSP 核心 | ✅ STFT / VAD×2 / DSP×3 / 指标 / RT60 估计 / 噪声平稳性判别 |
| Phase 3 数据配方 | ✅ V1 三测试集 notebook 已就绪：DNS 客观质量 / AISHELL 受控 CER / WenetSpeech 真实 CER |
| Phase 4 模型与训练 | ✅ 代码与 ONNX 流式导出完成 ｜ ☁️ 待用“高 SNR + 恒等样本”修复配方重训 |
| Phase 5 本地评测 | ✅ `rtse-eval` 按数据集能力自动选指标、支持动态分层与断点续跑 |
| Phase 6 Web 演示 | ✅ DSP 部分已实测验收；ASR 字幕待接入 |
| Phase 7 C++ 实时版 | ⏸ 需先装 CMake + MSVC（见 ISSUES.md I-03） |

> ⚠️ 当前仓库中的 DNS 训练模型是**修复训练分布之前的诊断基线**：能改善有噪代理样本，
> 但对干净输入过抑制，并使 WenetSpeech CER 恶化。新配方已修正，必须重训后才能作为最终结果。

## V1 数据职责

完整的数据来源、隔离规则和指标前提见 [docs/DATASETS_V1.md](docs/DATASETS_V1.md)。

| 用途 | 数据 | 指标 |
|---|---|---|
| 训练 | DNS5 `read_speech` 小子集 + 真实 DNS 噪声/RIR，在线混音 | 训练/验证损失 |
| 声学质量 | DNS5 留出语音 + 留出噪声 + RT60 匹配的真实/合成 RIR | SI-SDR、STOI、ESTOI、PESQ、DNSMOS |
| 受控中文识别 | AISHELL-1 test + 同一噪声/RIR 矩阵 | CER + 声学指标 |
| 真实会议泛化 | WenetSpeech `test_meeting` 原始录音，不二次退化 | 增强前后 CER |

---

## 已经能跑出来的数字

> 这一节只保留**与数据集无关**的数字（复杂度与实时性，只取决于模型结构和硬件）。
> 质量指标（SI-SDR / STOI / PESQ / CER）等新数据跑完再填，见
> [docs/METRICS.md](docs/METRICS.md)。

**测量条件**：i7-14700，CPU **单线程**，16 kHz / n_fft 512 / hop 256

| 方法 | 参数量 | RTF | p99 延迟 | 帧预算 | 余量 |
|---|---|---|---|---|---|
| wiener | 0 | 0.0058 | 0.42 ms | 16 ms | 38× |
| mmse-lsa | 0 | 0.0103 | 0.55 ms | 16 ms | 29× |
| crn-nano (ONNX) | 118 K | 0.0226 | 0.63 ms | 16 ms | 26× |
| crn-lite (ONNX) | 569 K | 0.0344 | 1.02 ms | 16 ms | 16× |

算法延迟固定 **32 ms**（STFT 窗长决定，CPU 再快也消不掉）。

更多见 [docs/METRICS.md](docs/METRICS.md)。

---

## 文档

| 文档 | 内容 |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | **总体方案**：架构、选型理由、8 个阶段与验收标准、风险预案 |
| [docs/DATASETS_V1.md](docs/DATASETS_V1.md) | **V1数据证据链**：来源、隔离、分层、指标前提与结论权限 |
| [notebooks/README.md](notebooks/README.md) | **数据设计** —— 用了哪些数据集、为什么、测试集怎么分层 |
| [docs/SETUP_NEW_MACHINE.md](docs/SETUP_NEW_MACHINE.md) | **换机器接手** —— 从零跑起来需要什么、缺什么、去哪拿 |
| [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) | **Colab 侧操作手册** —— 你要执行的部分 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 完成记录（只记已验证的事实） |
| [docs/ISSUES.md](docs/ISSUES.md) | 问题记录（含根因、证据与解法） |
| [docs/FINDINGS.md](docs/FINDINGS.md) | **实验发现** —— 违反直觉、值得深挖的现象 |
| [docs/METRICS.md](docs/METRICS.md) | 指标定义、测量方法学、结果表 |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | 本机环境勘察与由此产生的技术决策 |

---

## 分工

- ☁️ **Colab**：数据下载、数据合成、模型训练、ONNX 导出、PESQ 计算
- 🏠 **本地**：推理运行时、指标评测、RTF/延迟测量、ASR 对比、Web 演示

分工的**真正理由**是本地 C: 盘只剩约 16 GB，装不下几十 GB 的数据集
（GPU 其实够训这个规模的模型）。详见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

训练代码写成**位置无关**：同一份 `src/rtse/` 在两边跑同样的逻辑，
notebook 只负责下载数据和调用。将来想改成本地训练，不用改任何模型或训练代码。

---

## 三个贯穿全项目的工程约束

这三条是所有设计取舍的来源，破坏任何一条，指标就不可信：

1. **严格因果** —— 不用 center padding，不用双向 RNN。
   否则会出现"离线指标很好、上线就崩"，而且离线完全看不出来。
   已由 `test_model_is_strictly_causal` 强制。

2. **离线结果就是流式结果** —— `StreamingEnhancer.process()` 内部逐帧调用
   `process_frame`，离线评测与实时演示走的是**同一条代码路径**。
   一致性由结构保证，不靠人工对齐。

3. **NN 与 DSP 走完全相同的管线** —— `OnnxEnhancer` 包装成 `StreamingEnhancer` 后，
   与谱减、维纳在所有下游代码里完全等价，零特判分支。
   这保证了"NN vs DSP"的对比是公平的。

---

## 项目结构

```
src/rtse/
├── audio/      STFT/iSTFT（完美重构 + 流式一致 + dBFS 标定）、读写、重采样
├── vad/        能量+谱平坦度（自研）、WebRTC
├── dsp/        谱减、维纳、MMSE-LSA、MCRA 噪声估计、RT60 估计、噪声平稳性判别
├── models/     CRN-Lite 因果流式增强网络（本地与 Colab 共用）
├── data/       在线混音 + DNS/AISHELL/WenetSpeech 三套固定评测集生成
├── train/      损失、训练循环、ONNX 流式导出与三层校验
├── runtime/    逐帧管线、延迟计量、ONNX 适配器
├── metrics/    SI-SDR/SDR/SegSNR（自研）、STOI/ESTOI、指标注册表
├── server/     FastAPI + WebSocket
└── cli/        doctor / server / enhance / eval / asr
web/            前端（原生 ES Module + Canvas，零打包步骤）
notebooks/      Colab notebook × 3
```
