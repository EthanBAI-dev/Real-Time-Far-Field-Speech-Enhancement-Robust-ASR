# 完成记录

> 规则：每完成一个可验证的产出就在这里追加一条。**只记已验证的事实，不记计划。**
> 格式：`日期 | 阶段 | 做了什么 | 怎么验证的`

---

## Phase 0 — 文档与骨架 🏠本地

### 2026-08-05

- **环境勘察完成** → [`ENVIRONMENT.md`](ENVIRONMENT.md)
  - 验证方式：PowerShell 实测 CPU / 内存 / GPU / 磁盘 / 工具链，均为实际输出而非推测
  - 关键结论 3 条：① 系统无 Python，仅有 uv；② C: 仅剩 18.1 GB；③ 无 CMake / MSVC
- **总体方案定稿** → [`PLAN.md`](PLAN.md)
  - 含 8 个阶段、验收标准、风险预案、Colab/本地分工表

---

## Phase 1 — 环境搭建 🏠本地 ✅ **已验收**

### 2026-08-05

- **Python 3.11.15 环境就绪**（uv 独立安装，未污染系统）
  - 绕开了"系统无 Python"（I-01）和"uv 自带 3.14 过新"（I-02）两个坑
- **依赖全部装成**，`.venv` 体积 0.87 GB
  - torch 2.13.0**+cpu**（强制走 pytorch-cpu 索引，比 CUDA 版省约 2.5 GB 磁盘）
  - onnxruntime 1.28.0 / onnx 1.22.0 / faster-whisper 1.2.1 / ctranslate2 4.8.1
  - webrtcvad-wheels 2.0.14（官方 webrtcvad 无 Windows wheel，换带预编译的 fork）
  - 唯一装不上的是 `pesq`（I-05），已降级为 Colab 侧指标
- **`rtse-doctor` 全部 11 项实测检查通过**
  - 验证方式：每项都做**真实功能验证**而非仅 import ——
    soundfile 真写真读比对量化误差、onnxruntime 真导出真推理、
    soxr 真重采样后查主峰频率、webrtcvad 真判静音/纯音
  - 剩余 2 项告警均为预期：PESQ/DNSMOS 未就绪、Colab 产物尚未产出
- **确认 onnxruntime 只有 CPU provider** —— 这正是想要的，RTF/延迟必须在 CPU 单线程下测

---

## Phase 2 — 信号与 DSP 核心 🏠本地（进行中）

### 2026-08-05

- **STFT/iSTFT 完成并通过 23 项测试** → [`src/rtse/audio/stft.py`](../src/rtse/audio/stft.py)
  - ✅ **完美重构**：7 类信号（噪声/纯音/扫频/脉冲/静音/非整帧长/超短）误差均 < 1e-10，
    实测 1.8e-15
  - ✅ **严格因果**：不用 center padding，杜绝"离线指标好、上线就崩"
  - ✅ **流式/离线逐样本一致**：往返误差 1.6e-15
  - ✅ 反证测试锁死"对称 Hann 不满足 COLA"这个坑
  - ⚠️ 过程中发现并修复 I-06：sqrt-Hann 只在 50% 重叠时增益为 1，已泛化到任意 hop
- **音频 I/O 完成** → [`src/rtse/audio/io.py`](../src/rtse/audio/io.py)
  - 统一入口约定：16 kHz / 单声道 / float64 / [-1,1]
  - 修复 I-07：削波保护会静默改幅度，已改为返回增益并写死"指标只在内存算"的规矩
- **指标层骨架完成** → `src/rtse/metrics/`
  - SI-SDR / SDR / SegSNR 自研实现；STOI / ESTOI 走 pystoi
  - 指标注册表支持**缺指标降级**：装不上的指标标 n/a 并写明原因，不让评测崩掉

- **DSP 三件套完成** → [`src/rtse/dsp/`](../src/rtse/dsp/)
  - MCRA 噪声估计（最小值控制递归平均）—— **不依赖"前 N 帧是纯噪声"这个假设**，
    测试验证：信号一上来就是语音时仍能在 ~2 秒内收敛到真值 ±6 dB 内
  - 谱减法（Berouti 自适应过减）、维纳滤波（决策导向先验 SNR）、MMSE-LSA（Ephraim-Malah）
  - **架构决定：离线处理由流式实现驱动**，从结构上保证两者一致，
    测试验证逐样本偏差 < 1e-12
  - 修复 I-08 类问题：决策导向的反馈原本用未限幅增益，与实际施加的增益不一致，已改
- **VAD 三档完成** → [`src/rtse/vad/`](../src/rtse/vad/)
  - 自研能量+谱平坦度 VAD（自适应本底、迟滞、挂起）、WebRTC VAD
  - 解决 I-09：WebRTC 只接受 10/20/30 ms 帧，与本项目 16 ms 帧移不兼容，加了适配层
- **实时管线完成** → [`src/rtse/runtime/pipeline.py`](../src/rtse/runtime/pipeline.py)
  - 内建延迟计量：区分**算法延迟 / 处理延迟 p50·p95·p99 / 帧预算**三个层次
  - 修复 I-10：VAD 门控硬切把连续语音削掉三成能量，改为非对称平滑（快开慢关）
- **合成数据模块完成** → [`src/rtse/data/synth.py`](../src/rtse/data/synth.py)
  - 8 类噪声、**镜像源法 RIR**（不是假的指数衰减白噪声）、按活跃段定义的 SNR 混音
  - 测试验证：混音 SNR 误差 < 0.5 dB，RIR 的 Schroeder T60 与目标吻合
- **测试全绿：61 项通过**

---

## Phase 6 — Web 演示 🏠本地 ✅ **DSP 部分已验收**

### 2026-08-05

- **FastAPI 服务 + 三个页面上线** → [`src/rtse/server/`](../src/rtse/server/)、[`web/`](../web/)
  - `/` 实时麦克风演示、`/lab` 文件实验台、`/bench` 基准看板
  - 前端零打包步骤（原生 ES Module + Canvas），一条 `uv run rtse-server` 就能起
- **实测验证（浏览器 + WebSocket 客户端双路）**：
  - 文件实验台端到端跑通，真实数字：babble @ 5 dB 下
    ΔSI-SDR 分别为 specsub +1.77 / wiener +1.87 / mmse-lsa +2.13
  - **WebSocket 实时路径与离线管线数值一致**，最大偏差 5e-8
    （正好是 float32 传输精度，说明服务端逻辑零差异）
  - 实时性：p99 处理延迟 **0.16 ~ 0.59 ms**，对 16 ms 帧预算有 27~100 倍余量
  - 方法热切换正常，输出样本数与输入精确相等
  - 频谱图渲染验证：像素分布 p10=14 / p50=90 / p90=178，动态范围完全打开
- **过程中解决 3 个环境/可视化问题**：
  - I-11 控制台代码页是 **cp932（日文）**，print 中文直接崩溃
  - I-12 频谱图洗白 —— dB 参考点未定义，已建立 dBFS 标定并加回归测试
  - I-13 改了 Python 没重启服务，白排查一轮
- **尚未接入**：ASR 字幕（需 Whisper 模型）、神经网络模型（需 Colab 产出）

### 2026-08-05（补：无麦克风环境下的可用性）

- **确认本机没有可用的麦克风端点**（I-16）：
  Realtek 麦克风插孔未接入，Windows 隐藏了该端点，
  浏览器 `enumerateDevices()` 的 `audioinput` 数量为 **0**。
  隐私设置是 `Allow`，不是权限问题。
- **新增「虚拟麦克风」** —— 实时演示页在没有麦克风时依然完整可用
  - 服务端按指定条件（噪声/SNR/T60）合成音频，前端按**真实时间节奏**
    （每 128 ms 一块）推进**同一条** WebSocket 路径
  - 不是"离线处理"：按真实节奏推送，所以 RTF 与 p95/p99 与真麦克风**等价**
  - 合成放服务端，保证"0 dB"在虚拟麦克风、文件实验台、Colab 三处是同一个定义
  - 检测到无输入设备时自动切换并说明原因
- **修正误导性报错**（I-16）：原本不分错误类型一律提示"只在 localhost 或 HTTPS 下可用"，
  对 `NotFoundError` 是完全错误的引导。现在按 `DOMException.name` 逐一准确诊断。
- **修复静态资源缓存**（I-17）：`StaticFiles` 不发 `Cache-Control`，
  浏览器启发式缓存导致改了前端刷新没反应（新 HTML + 旧 JS，症状极具迷惑性）。
  已加中间件统一发 `no-cache, must-revalidate`。
- **修复电平读数摆动**（I-18）：瞬时 RMS 摆动十几 dB，加了 1.6 s 的 dB 域平滑。
- **实测验证（虚拟麦克风，keyboard @ 0 dB，mmse-lsa）**：

    | 指标 | 实测 |
    |---|---|
    | RTF | 0.009 ~ 0.021 |
    | 处理延迟 p95 | 0.30 ms |
    | 处理延迟 p99 | 0.39 ms（帧预算 16 ms） |
    | 净衰减（平滑后） | 6.7 ~ 9.3 dB |
  - 方法热切换 A/B 正常：none 0.3 dB → specsub 4.5 → wiener 5.5 → mmse-lsa 16.3（瞬时）
  - 频谱图肉眼可见降噪；**键盘冲激在增强输出里依然清晰可见** ——
    F-02 的失效模式在实时演示里被直接看到了

---

## Phase 4 — 模型与训练代码 🏠本地已完成，☁️Colab 待执行

### 2026-08-05

> 说明：**训练本身在 Colab**，但训练所需的全部代码在本地写完并验证。
> 这样做的收益是：Colab 上只需调用，不需要调试 —— 训练几小时后才发现导不出来是最糟的情况。

- **CRN-Lite 模型完成** → [`src/rtse/models/crn.py`](../src/rtse/models/crn.py)
  - 自行定义（不引用外部模型仓库），三档预设：

    | 模型 | 参数量 | ONNX 大小 |
    |---|---|---|
    | crn-nano | 118,370 | 475 KB |
    | crn-lite | 569,282 | 2,237 KB |
    | crn-large | 1,770,722 | — |
  - 输出**有界复数比值掩码（CRM）**，同时修正实部虚部 ——
    这是它有可能超越 MMSE-LSA 的根本原因（DSP 方法只能沿用带噪相位）
  - ✅ **严格因果性已由测试强制**：改动第 t 帧之后的输入，
    第 t 帧及之前的输出一个比特都不变
  - ✅ **流式与整段推理数值一致**（偏差 < 1e-5）
- **训练代码完成** → [`src/rtse/train/`](../src/rtse/train/)
  - 三项组合损失：压缩谱复数 MSE + 多分辨率 STFT + SI-SDR，各分项单独记录
  - 断点续训**含优化器动量、学习率调度、随机数状态** —— Colab 断线是常态
  - warmup + 余弦退火、梯度裁剪、AMP、坏 batch 跳过
- **在线混音数据集完成** → [`src/rtse/data/dataset.py`](../src/rtse/data/dataset.py)
  - 同时支持目录扫描与文件列表（Colab 用按说话人划分好的清单）
  - 修复 I-15：随机增益的削波保护有 bug，会把削波数据喂给模型
  - 解决 I-14：torch 与 numpy 的 STFT 帧数不一致，尾部 256 样本重建不出来
- **✅ ONNX 流式导出打通并三层校验通过**（用随机权重模型在本地验证）
  → [`src/rtse/train/export.py`](../src/rtse/train/export.py)

    | 校验项 | crn-nano | crn-lite |
    |---|---|---|
    | ONNX流式 vs PyTorch流式 | 1.07e-06 | 1.19e-06 |
    | **ONNX流式 vs PyTorch整段** | 8.35e-07 | 1.25e-06 |
    | 相对误差 | 2.28e-07 | 3.19e-07 |
    | 状态形状稳定 | ✅ | ✅ |
    | **判定** | **PASS** | **PASS** |
  - 导出的是**单帧步进图**，5 个卷积缓存 + 1 个 GRU 隐状态显式外置为图的输入输出
  - **ONNX 比 PyTorch 快得多**：crn-nano 0.104 vs 0.520 ms（5.0×），
    crn-lite 0.382 vs 0.666 ms（1.7×）—— 直接支持了"导出 ONNX 而非部署 PyTorch"的决定
- **神经模型已接入现有 Pipeline** → [`src/rtse/runtime/onnx_enhancer.py`](../src/rtse/runtime/onnx_enhancer.py)
  - 包成 `StreamingEnhancer` 后，与 DSP 方法在所有下游代码里**完全等价**，
    零特判分支。这保证了 "NN vs DSP" 的对比是公平的（同一条路径、同样的计时点）
  - 实测 crn-lite 完整管线：RTF 0.0344，p99 **1.02 ms**（帧预算 16 ms，16 倍余量）
  - Web 界面自动发现 `models/*.onnx`，训练完放进去就能用，不用改代码
- **测试全绿：107 项通过**

---

## Colab 侧交付物 ☁️ **已就绪，待你执行**

### 2026-08-05

- **三个 notebook 生成并通过语法校验** → [`notebooks/`](../notebooks/)
  - `01_data_prep.ipynb`（10 code / 6 md）：下载 THCHS-30 + MUSAN + RIRS，
    **按说话人划分**，补充合成冲激噪声，合成固定测试集并打包
  - `02_train.ipynb`（6 code / 4 md）：三档模型训练，断点续训，训练曲线
  - `03_export_eval.ipynb`（7 code / 5 md）：ONNX 导出 + 三层校验 + PESQ + DNSMOS 下载
  - 每个 notebook 前三 cell 固定：挂载 Drive → 安装代码 → **自检**
    （验证 Colab 与本地是同一条信号链路，STFT 与 dBFS 标定完全一致）
- **打包脚本** → [`scripts/pack_for_colab.py`](../scripts/pack_for_colab.py)
  - 实测产出 112 KB / 46 个文件，白名单机制（不会误传 data/ models/ .venv/）
- **操作手册** → [`COLAB_GUIDE.md`](COLAB_GUIDE.md)，含免费版 T4 + 2 vCPU 的实操建议

### 2026-08-05（补：适配实际 Drive 路径 + 修掉一个致命性能问题）

- **适配带空格的 Drive 路径**（用户实际路径是 `MyDrive/Audio AI/RTSE`）
  - 审计出 3 处 shell 命令的插值没加引号，全部改用 `shlex.quote`
  - 用真实带空格的路径实跑验证了配置逻辑与目录创建
  - 配置 cell 增加断言：目录不存在时直接报错并列出该目录现有文件
- **新增 `hybrid` 数据模式并设为默认**
  - 压缩包留 Drive（一次下载永久有效），每会话解压到本地盘
  - 理由：Drive 是 FUSE 挂载，创建文件开销极高。THCHS-30 有一万多个小 wav，
    直接解压到 Drive 可能要几小时，解压到本地盘只要几分钟
  - 下载与解压用两个独立标记，新会话只解压不重下
- **✅ 修复 I-19：卷混响用直接卷积，会让训练彻底卡死**
  - `np.convolve` 在 T60=0.6 s 时**单样本要 6996 ms**，改用 `fftconvolve` 后 2.0 ms，
    **快 3482 倍**，数值等价
  - 修复前按 `reverb_prob=0.5` 估算，一个 20000 样本的 epoch
    光数据加载就要 **约 5 小时**，GPU 全程空转，且**不会有任何报错**
  - 修复后实测：单样本 5.1 ms，2 worker 393 样本/秒，epoch 加载 0.8 分钟
  - 加了两条回归测试：性能门限（< 50 ms）+ 与直接卷积的数值等价
  - **发现契机**：用户问「免费版该选哪个 GPU」→ 去核对 2 vCPU 能否喂饱 GPU → 撞上
- **测试增至 109 项，全部通过**
- **指标表框架** → [`METRICS.md`](METRICS.md)，已实测部分填入真实数字，待填部分明确标注

---

## Phase 5 — 本地推理与评测 🏠本地（进行中）

### 2026-08-07

- **首批真实训练结果落地**：Colab 训练 60 epoch（crn-nano/lite/large，THCHS-30），
  导出全部 PASS，780 样本完整测试集本地跑通，`rtse-doctor` 全绿
- **发现并部分修复 MCRA 高 SNR bug**（[`ISSUES.md`](ISSUES.md) I-20）：
  - 复核用户贴出的汇总结果时发现 DSP 三方法 SI-SDR 全部低于不处理，追查后确认
    是 SNR≥15 dB 时噪声估计系统性失真（最严重达 22dB / 170倍误差），根因是
    混响 + 真实连续语音的静音过于稀少，2 秒单窗口的最小值追踪找不到真实地板
  - 系统性网格搜索 delta/alpha_s 证明**不是参数问题**（即使把 delta 从 5.0
    压到 1.5 依然有 15dB 误差，且开始损害低 SNR 准确度）
  - 修复方案：双时间尺度最小值跟踪（原有 2 秒短窗口 + 新增 16 秒长窗口并行，
    取更低者，数学上保证只会更好不会更差）
  - 用真实测试数据验证：10 随机样本均值 2.32→4.17dB，最严重案例
    -4.79→+8.01dB；全量 780 样本本地重算，wiener 从「排除高SNR才正常」变成
    全 SNR 区间连贯为正
  - **诚实记录未完全解决的部分**：短促静音间隙+极高SNR的合成边界场景依然失败，
    确认不是 alpha_d 收敛速度问题（调低反而更差），需要比调参更深的改动
    （IMCRA 式软判决）才能彻底解决——加了一条数学不变量测试（保证安全）+
    保留原 xfail 测试（如实记录残留问题）
- **`METRICS.md`/`FINDINGS.md` 用修复后的真实数字更新**：
  - crn-nano 修复前"看起来赢过" DSP 基线，修复后实际**低于** specsub/wiener/mmse-lsa——
    记录为一条方法学教训：先确认基线没有被测量问题低估，再下结论
  - F-05（NN vs DSP，非平稳噪声上优势明显）、F-06（T60≥0.6s 全员负分，去混响
    模块缺失的代价）两条新发现，均用真实 780 样本数据支撑
- **测试增至 111 项（110 passed + 1 xfailed），全部符合预期**

<!-- 后续条目在此追加 -->
