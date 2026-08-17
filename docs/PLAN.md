# 实时远场语音增强与 ASR 鲁棒性系统 —— 总体方案

> 项目代号：**RTSE**（Real-Time Speech Enhancement）
> 本机路径：`C:\Users\haku\Documents\trae_projects\VAD`
> 制定时间：2026-08-05

---

## 0. 一句话目标

做出一条**可实时运行、可量化、可演示**的语音前端链路：

```
麦克风 → VAD → 去混响 → 降噪（DSP 基线 vs 神经网络）→ [可选波束形成] → ASR → 实时字幕
```

并用一份完整的指标矩阵回答一个业务问题：**语音增强到底让 ASR 的字错率降低了多少，代价是多少 ms 延迟和多少 CPU。**

---

## 1. 分工：Colab 做什么，本地做什么

这是本方案的核心约束，所有设计都围绕它展开。

| 阶段 | 在哪里 | 理由 |
|---|---|---|
| 数据下载（DNS / RIR / MUSAN / AISHELL） | **Colab** | 本地 C: 仅 18 GB 可用 |
| 数据合成（加噪 + 卷混响 + 分段） | **Colab** | 紧挨数据，避免来回传输 |
| 模型训练 | **Colab** | 用户指定；同时规避本地磁盘瓶颈 |
| ONNX 导出 | **Colab**（训练完顺手导出） | 产物只有几 MB |
| 推理 / 流式运行时 | **本地** | |
| 指标评测（SI-SDR / STOI / DNSMOS / PESQ） | **本地** | 测试集只需几百 MB |
| ASR CER / WER 对比 | **本地** | |
| RTF / 延迟测量 | **本地** | 必须在真实目标机器上测才有意义 |
| Web 可视化 Demo | **本地** | |
| C++ 实时版 | **本地** | |

### 跨越边界的三样东西（体积都很小）

1. **代码**：`src/rtse/` 整个包 → Colab（一条 `pip install -e` 或直接 clone）
2. **模型**：Colab → 本地，`.onnx` + `.pt`，单个 < 20 MB
3. **测试集**：Colab 合成后打包 → 本地，目标 **< 2 GB**（含干净参考、带噪输入、转写文本）

> 三样东西的传输方式：Google Drive。详见 `COLAB_GUIDE.md`。

---

## 2. 目录结构

```
VAD/
├── README.md                    项目门面：能力清单 + 关键指标 + 一键运行
├── pyproject.toml               uv 管理的依赖与入口点
├── docs/
│   ├── PLAN.md                  ← 本文件
│   ├── ENVIRONMENT.md           本机环境勘察
│   ├── PROGRESS.md              完成记录（每完成一项就追加）
│   ├── ISSUES.md                问题记录（每遇到一个坑就追加）
│   ├── COLAB_GUIDE.md           Colab 侧操作手册
│   └── METRICS.md               指标定义 + 结果表
├── configs/                     YAML 配置（管线 / 模型 / 评测矩阵）
├── src/rtse/
│   ├── audio/                   读写、重采样、分帧、STFT/iSTFT（含完美重构校验）
│   ├── vad/                     能量VAD / 谱平坦度VAD / Silero-ONNX
│   ├── dsp/                     谱减法、维纳滤波、MMSE-LSA、噪声估计(MCRA)
│   ├── dereverb/                单通道 WPE
│   ├── beamform/                延迟求和 / MVDR / GEV（Phase 8）
│   ├── models/                  CRN-Lite 流式增强网络（PyTorch，本地与 Colab 共用）
│   ├── data/                    混音合成、清单(manifest)、Dataset
│   ├── train/                   训练循环、损失函数、ONNX 导出（Colab 调用）
│   ├── runtime/                 ONNX 流式会话、环形缓冲、逐帧处理器、延迟计量
│   ├── asr/                     faster-whisper 封装、中英文本归一化、CER/WER
│   ├── metrics/                 SI-SDR / STOI / PESQ / DNSMOS
│   ├── eval/                    批量评测矩阵 → results/*.json
│   └── server/                  FastAPI：REST + WebSocket 音频流
├── web/                         前端（原生 ES Module + Canvas，无打包步骤）
├── notebooks/                   Colab notebook（数据 / 训练 / 导出）
├── scripts/                     本地命令行入口
├── models/                      .onnx 产物（git 忽略）
├── data/                        本地测试集（git 忽略）
├── results/                     评测输出、图表（git 跟踪 json，忽略音频）
└── tests/                       pytest 单元测试
```

---

## 3. 技术选型与理由

### 3.1 信号基础

| 参数 | 取值 | 理由 |
|---|---|---|
| 采样率 | 16 kHz 单声道 | ASR 标准；DNS Challenge 与 AISHELL 均为 16k |
| 窗长 / 帧移 | 512 / 256（32 ms / 16 ms） | 频率分辨率 31.25 Hz，兼顾语音谐波与低延迟 |
| 窗函数 | sqrt-Hann（分析与合成同窗） | 满足 COLA，OLA 完美重构 |
| 算法延迟 | 32 ms（一个窗） | 全部模块保持因果，不引入额外前瞻 |

**硬性要求**：`tests/test_stft.py` 必须验证 STFT→iSTFT 重构误差 < 1e-6，这是整条链路的地基。

### 3.2 VAD

三档实现，用来做对比而不是选一个：

1. **能量 + 谱平坦度**（自研）：展示 DSP 功底，含自适应噪声本底跟踪与迟滞判决（hangover）
2. **WebRTC VAD**（`webrtcvad-wheels`）：工业界事实基线
3. **Silero VAD ONNX**（约 2 MB）：当前效果最好的轻量方案，作为上界参考

评测：在合成集上算 VAD 的**帧级 F1 / 误触发率 / 漏检率 vs SNR**，并展示 VAD 门控对下游 ASR 的影响
（错误切断语音会直接吃掉字符）。

### 3.3 降噪：DSP 基线

必须先做传统方法，这是简历里"证明会做信号处理而不只是调包"的部分：

| 方法 | 要点 |
|---|---|
| 谱减法 | 过减因子 α、谱下限 β、半波整流；重点展示并解决**音乐噪声** |
| 维纳滤波 | 基于先验/后验 SNR，**决策导向（Decision-Directed）** 先验 SNR 估计 |
| MMSE-LSA | Ephraim-Malah 对数谱幅度估计，音乐噪声显著优于谱减 |
| 噪声估计 | **MCRA / IMCRA**（最小值控制递归平均），不假设"前 N 帧是纯噪声" |

> 失败案例分析里必须包含：谱减法在非平稳噪声下的音乐噪声频谱图对比。

### 3.4 去混响

**单通道 WPE**（加权预测误差），迭代式，帧级因果近似版本用于流式。
远场是本项目的关键词，只做降噪不做去混响，"远场"就名不副实。

### 3.5 降噪：神经网络

**主模型：CRN-Lite**（自研，仓库内完整定义，不依赖任何外部模型仓库）

```
输入  : 带噪复数谱 (2, T, F=257)，幅度做 0.3 次幂压缩
编码器: 5 层因果 2D 卷积，频率轴 stride 2，通道 16→32→32→64→64
瓶颈  : 2 层 GRU（沿时间轴，天然因果流式）
解码器: 5 层转置卷积 + 跳跃连接（对称）
输出  : 有界复数比值掩码 CRM（tanh 限幅）
参数量: 目标 < 1.0 M
```

**因果性是硬约束**：时间轴只做左侧 padding，GRU 单向。任何一处双向或未来帧，
流式导出就会失效，整个 RTF/延迟指标全部作废。

**损失函数**（三项加权）：
1. 幂律压缩谱的复数 MSE（DNS 基线的标准做法，对听感友好）
2. 多分辨率 STFT 幅度损失
3. 时域 SI-SDR

**对比模型**：
- 更小：`CRN-Nano`（< 100 K 参数），用来占据"极低算力"那一档
- 更大：`DPCRN`（双路径 RNN，intra-freq + inter-time），用来占据"高性能"那一档

三档模型 + 三个 DSP 基线 = 六条曲线，指标表才有说服力。

### 3.6 ONNX 流式导出

这是本项目工程含量最高的一环，也是最容易做错的一环。

- 导出的不是"整段音频"图，而是**单帧步进图**：
  `(frame_in, conv_cache_in, gru_state_in) → (frame_out, conv_cache_out, gru_state_out)`
- 卷积的时间维缓存显式外置为输入/输出张量（因为 ONNX 图是无状态的）
- 导出后必须做**流式 vs 整段一致性校验**：逐帧推理结果与整段推理结果误差 < 1e-4，
  写成 `tests/test_streaming_parity.py`。这条测试不通过，后面所有实时指标都不可信。

### 3.7 ASR

- **引擎**：faster-whisper（CTranslate2 后端，CPU int8 可跑，GPU 可选）
- **中文**：主战场。测试集 AISHELL-1 test（约 1.2 GB，7176 条，开放下载）
- **英文**：LibriSpeech test-clean 子集，作为跨语言佐证
- **CER 计算**：中文按字符级。文本归一化必须做全：
  全角半角、标点剥离、中文数字↔阿拉伯数字、英文大小写、空格处理。
  归一化不做干净，CER 数字就是噪声。

### 3.8 指标（简历里要出现的那一组）

| 类别 | 指标 |
|---|---|
| 客观语音质量 | SI-SDR、PESQ (WB)、STOI / ESTOI |
| 无参考 MOS | DNSMOS P.835（SIG / BAK / OVRL） |
| 下游任务 | **ASR CER（中）/ WER（英），增强前 vs 增强后** |
| 复杂度 | 参数量、MACs、模型文件大小 |
| 实时性 | **RTF**（单线程 CPU）、单帧处理延迟 p50/p95/p99、算法延迟、端到端延迟 |
| 鲁棒性切片 | SNR × 噪声类型 × 混响 T60 三维矩阵 |
| 对比 | 传统 DSP vs 神经模型，逐格对比 |
| 失败案例 | 至少 5 个，带频谱图与成因分析 |

**评测矩阵设计**：

- SNR：`{-5, 0, 5, 10, 15, 20}` dB
- 噪声类型：`{babble, cafeteria, car, keyboard, music, white, street}`
- 混响：`{anechoic, T60≈0.3s, T60≈0.6s, T60≈0.9s}`
- 方法：`{noisy, specsub, wiener, mmse-lsa, wpe+wiener, crn-nano, crn-lite, dpcrn, crn-lite+wpe}`

完整笛卡尔积 = 6×7×4×9 = 1512 格。**不全跑**：
主表跑 SNR × 方法（固定噪声混合、T60=0.3），副表各自单变量扫描。

### 3.9 Web 可视化

**技术选择：FastAPI + 原生前端（ES Module + Canvas），零打包步骤。**

理由：本机虽有 Node，但引入 Vite/React 会带来构建步骤、依赖体积和离线可用性问题。
频谱图本来就要用 Canvas 手绘，图表库省不了多少事。一条 `uv run rtse-server` 就能起，
对"给别人演示作品集"这个场景是最优解。

三个页面：

**① 实时麦克风演示** `/live`
- AudioWorklet 采集 → 16 kHz 单声道 → WebSocket 二进制帧 → 服务端逐帧管线 → 回传
- 可视化：上下双频谱图（带噪 / 增强，实时滚动）、VAD 门控时间轴、输入输出电平表
- 实时数码管：RTF、单帧延迟 p95、端到端延迟、当前 SNR 估计
- 实时字幕区（ASR 结果流式追加）
- 开关面板：VAD 类型 / 去混响开关 / 降噪方法（DSP vs 各神经模型）**热切换**，
  切换时字幕与频谱同屏对比

**② A/B 文件实验台** `/lab`
- 上传或选择测试集样本 → 一次跑多种方法 → 并排波形 + 频谱 + 播放器
- 下方即时指标表（SI-SDR / STOI / DNSMOS / CER），差值高亮

**③ 基准看板** `/bench`
- 读取 `results/*.json` 渲染完整指标矩阵
- SNR-CER 折线、方法对比雷达、RTF vs 质量散点（帕累托前沿）
- 失败案例画廊

---

## 4. 阶段划分与验收标准

每个阶段有明确的**可验证产出**，做完就写进 `PROGRESS.md`。

### Phase 0 — 文档与骨架 🏠本地
- [ ] 目录结构建立
- [ ] `PLAN.md` / `ENVIRONMENT.md` / `PROGRESS.md` / `ISSUES.md` 就位
- **验收**：本文件存在且被认可

### Phase 1 — 环境搭建 🏠本地
- [ ] uv 拉取 Python 3.11，创建 `.venv`
- [ ] 安装依赖（torch-cpu / onnxruntime / soundfile / sounddevice / fastapi / librosa / pystoi / pesq / jiwer / faster-whisper ...）
- [ ] 冒烟测试：能读写 wav、能跑 torch、能起 onnxruntime、能列出麦克风设备
- **验收**：`uv run python -m rtse.doctor` 全绿

### Phase 2 — 信号与 DSP 核心 🏠本地
- [ ] STFT/iSTFT 完美重构（测试通过）
- [ ] 三种 VAD
- [ ] 谱减 / 维纳 / MMSE-LSA / MCRA 噪声估计
- [ ] 单通道 WPE
- [ ] 指标模块（SI-SDR / STOI / PESQ）
- [ ] 本地合成小样本（自制噪声）验证管线端到端跑通
- **验收**：`uv run rtse-enhance --method wiener in.wav out.wav` 出声且 SI-SDR 提升为正

### Phase 3 — 数据配方 ☁️Colab
> **2026-08-07 数据集定稿**（此前基于 THCHS-30 + MUSAN 的两版已废弃并删除）。
> 完整设计与理由见 [`notebooks/README.md`](../notebooks/README.md)。
- [x] `notebooks/01_data_prep.ipynb`（已写，**尚未在 Colab 实跑**）
- [x] 训练语音：DNS Challenge 4 `read_speech` 1 分片（≈21 小时，独立可解压）
- [x] 噪声：DNS `noise_fullband`（AudioSet + Freesound），**按平稳性自动分两组**
      （`rtse.dsp.stationarity`，DNS 不带这个标注，从信号本身算）
- [x] RIR：DNS 真实 IR + 本项目合成（镜像源法），测试集里**并行分层**
- [x] 中文 ASR 评测：WenetSpeech `test_meeting`（HF 镜像，无需填表）
- [x] 测试集分层：SNR{−5,0,5,10,15} × 噪声{稳态,非稳态} × RIR{合成 4 档 RT60, 真实}
- **验收**：本地能下载并加载测试集，`uv run rtse-eval` 跑通

### Phase 4 — 模型与训练 ☁️Colab
- [ ] `notebooks/02_train.ipynb`
- [ ] 三档模型训练至收敛，记录训练曲线
- [ ] `notebooks/03_export_onnx.ipynb`：导出流式 ONNX
- [ ] **流式一致性校验在 Colab 侧先过一遍**
- **验收**：`.onnx` 文件下载到本地 `models/`，本地一致性测试同样通过

### Phase 5 — 本地推理与评测 🏠本地
- [ ] ONNX 流式运行时（环形缓冲 + 状态管理）
- [ ] RTF / 延迟测量（单线程绑核，多次重复取分位数）
- [ ] 完整指标矩阵评测 → `results/`
- [ ] ASR CER 对比（增强前 vs 后 vs 干净上界）
- [ ] 失败案例挑选与分析
- **验收**：`docs/METRICS.md` 里的表格被真实数字填满

### Phase 6 — Web 演示 🏠本地
- [ ] FastAPI 服务 + WebSocket 音频通道
- [ ] AudioWorklet 前端采集
- [ ] 三个页面全部可用
- **验收**：浏览器打开、对着麦克风说话、看到频谱变化与字幕、能热切换方法

### Phase 7 — C++ 实时版 🏠本地（⚠️需先安装工具链）
**前置条件（需用户确认后安装）**：CMake、Visual Studio 2022 Build Tools、PortAudio、ONNX Runtime C++
- [ ] PortAudio 回调式低延迟管线
- [ ] ONNX Runtime C API 逐帧推理
- [ ] 与 Python 版数值对齐验证
- **验收**：`rtse_rt.exe` 实测端到端延迟并与 Python 版对比

### Phase 8 — 高级扩展（按需）
- [ ] AEC（回声消除，可复用 NLMS/FDAF/RLS 思路）
- [ ] 双麦 MVDR 波束形成
- [ ] 目标说话人提取
- [ ] 流式 ASR + Barge-in 打断

---

## 5. 风险清单与预案

| 风险 | 影响 | 预案 |
|---|---|---|
| Windows 上 `pesq` 无 wheel 需编译 | Phase 2 阻塞 | 优先找 wheel；实在不行降级为可选依赖，PESQ 在 Colab 侧算 |
| `sounddevice` / PortAudio 拿不到麦克风 | Phase 6 实时演示废掉 | 前端用浏览器 WebAudio 采集，服务端不碰硬件（**已采纳为默认方案**） |
| Colab 会话超时、数据丢失 | Phase 3/4 反复重来 | 数据与 checkpoint 一律写 Google Drive；训练支持断点续训 |
| ONNX 流式与整段不一致 | 所有实时指标作废 | 强制一致性测试；一旦不过就回退到"分块推理 + 重叠拼接" |
| C: 盘 18 GB 被撑爆 | 系统层面问题 | 硬预算 5 GB；模型/数据目录加 `.gitignore`；必要时迁 D: |
| faster-whisper GPU 依赖 cuDNN 缺失 | ASR 评测变慢 | 默认 CPU int8；GPU 作为可选加速，失败自动回退 |
| 中文 CER 归一化不彻底 | 指标不可信 | 单独写归一化单测，用手工构造的边界样例覆盖 |

---

## 6. 立刻可开始的部分

Phase 0 → 1 → 2 完全不依赖 Colab，可以现在就做完。
Colab 侧的 notebook 会在 Phase 2 完成后一并交付，届时用户只需在 Colab 上按 `COLAB_GUIDE.md` 执行。
