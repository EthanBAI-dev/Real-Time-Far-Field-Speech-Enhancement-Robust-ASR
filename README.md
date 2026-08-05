# RTSE

**Real-Time Far-Field Speech Enhancement for Robust ASR**
远场语音增强与 ASR 鲁棒性系统

[![tests](https://github.com/EthanBAI-dev/Real-Time-Far-Field-Speech-Enhancement-Robust-ASR/actions/workflows/tests.yml/badge.svg)](https://github.com/EthanBAI-dev/Real-Time-Far-Field-Speech-Enhancement-Robust-ASR/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)

```
麦克风 → VAD → 去混响 → 降噪(DSP 基线 vs 神经网络) → [波束形成] → ASR → 实时字幕
```

一条**可实时运行、可量化、可演示**的语音前端链路，用完整的指标矩阵回答：
**语音增强让 ASR 字错率降低了多少，代价是多少毫秒延迟和多少 CPU。**

---

## 快速开始

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
| Phase 2 信号与 DSP 核心 | ✅ STFT / VAD×2 / DSP×3 / 指标 / 合成数据 |
| Phase 3 数据配方 | ☁️ **notebook 已就绪，待在 Colab 执行** |
| Phase 4 模型与训练 | ✅ 代码完成 + ONNX 导出已验证 ｜ ☁️ 训练待执行 |
| Phase 5 本地评测 | ⏸ 待 Colab 产出测试集与模型 |
| Phase 6 Web 演示 | ✅ DSP 部分已实测验收；ASR 字幕待接入 |
| Phase 7 C++ 实时版 | ⏸ 需先装 CMake + MSVC（见 ISSUES.md I-03） |

**测试：107 项全部通过。**

---

## 已经能跑出来的数字

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
| [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) | **Colab 侧操作手册** —— 你要执行的部分 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 完成记录（只记已验证的事实） |
| [docs/ISSUES.md](docs/ISSUES.md) | 问题记录（15 条，含根因与解法） |
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
├── dsp/        谱减、维纳、MMSE-LSA、MCRA 噪声估计
├── models/     CRN-Lite 因果流式增强网络（本地与 Colab 共用）
├── data/       合成噪声/镜像源法 RIR/按活跃段的 SNR 混音、在线混音数据集
├── train/      损失、训练循环、ONNX 流式导出与三层校验
├── runtime/    逐帧管线、延迟计量、ONNX 适配器
├── metrics/    SI-SDR/SDR/SegSNR（自研）、STOI/ESTOI、指标注册表
├── server/     FastAPI + WebSocket
└── cli/        doctor / server / enhance / eval / asr
web/            前端（原生 ES Module + Canvas，零打包步骤）
notebooks/      Colab notebook × 3
```
