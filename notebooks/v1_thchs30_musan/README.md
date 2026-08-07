# v1 · THCHS-30 + MUSAN + RIRS_NOISES

第一版数据配方与训练流程。**已跑通并交付了首批真实结果**（crn-nano/lite/large，60 epoch），
是 [`docs/METRICS.md`](../../docs/METRICS.md) 里当前那批指标数字的来源。这个版本**保留不删**，
作为 [v2](../v2_dns_real_noise/README.md)（DNS Challenge 真实噪声）的对照基线——两版模型
最终会在同一份评测方法下并排比较，回答"换成真实录制噪声到底有没有用、有多少用"这个问题。

## 数据构成

| 组成 | 来源 | 体积 | 性质 |
|---|---|---|---|
| 干净语音 | THCHS-30（openSLR SLR18） | ~6.4 GB | 真实录制，带完整转写 |
| 训练噪声 | MUSAN noise 子集（SLR17） | ~11 GB | 真实录制 |
| 训练 RIR | RIRS_NOISES（SLR28） | ~6 GB | 真实测量 + 仿真混合 |
| 冲激型噪声（训练+测试） | 本项目 `rtse.data.synth.make_noise()` 程序合成 | 0 | **合成**，7 类（keyboard/cafeteria/hum/white/pink/car/babble） |
| **固定测试集的噪声** | 同上，`make_noise()` 合成 | — | **合成**，不是从 MUSAN 抽样 |

## 已知局限（正是 v2 存在的原因）

**固定测试集的噪声从一开始就是 100% 程序合成的**，从未使用过 MUSAN 里的真实录制噪声
——训练时在线混音确实用了真实 MUSAN 噪声，但用来算 SI-SDR/STOI/PESQ 这些最终指标的
`data/testset/`，噪声完全来自 `make_noise()` 的参数化生成器。这类合成噪声频谱结构规整、
统计特性单一，可能让指标好看程度超出真实场景。这一点在开发过程中一直存在，直到准备做
DNS Challenge 真实噪声版本时才被明确记录下来。

## 状态

- 训练完成：crn-nano / crn-lite / crn-large，60 epoch，checkpoint 见 Colab Drive
- 导出 ONNX：三档全部 PASS（流式一致性校验）
- 已知问题：[`docs/ISSUES.md`](../../docs/ISSUES.md) I-20（MCRA 高 SNR bug，已部分修复）、
  I-21（测试集参考文本比截断音频长，已修复但需要重新生成数据）

## 三个 notebook

1. `01_data_prep.ipynb` —— 下载语料、按说话人划分、合成固定测试集
2. `02_train.ipynb` —— 训练三档模型
3. `03_export_eval.ipynb` —— 导出流式 ONNX + Colab 侧补算 PESQ/DNSMOS

用法见 [`docs/COLAB_GUIDE.md`](../../docs/COLAB_GUIDE.md)。Drive 上的项目根固定用
`DRIVE_ROOT = '/content/drive/MyDrive/Audio AI/RTSE'`——v2 用的是另一个子目录，
两版数据/checkpoint 不会互相覆盖。
