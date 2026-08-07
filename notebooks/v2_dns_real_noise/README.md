# v2 · DNS Challenge 真实噪声

第二版数据配方，跟 [v1](../v1_thchs30_musan/README.md) 并排保留，作为可对比的迭代——
不是替换关系。用**同一套模型代码、同一套超参、同一个说话人/噪声/RIR 划分种子**，
只换一个变量：噪声和房间冲激响应从合成/MUSAN 换成 DNS Challenge 的真实录制数据。

## 起因

用户明确怀疑 v1 里合成噪声（`rtse.data.synth.make_noise()` 程序生成的 keyboard/
cafeteria/hum/white/pink/car/babble）的音质代表性，要求换用真实噪声。核查后发现问题
比预想的更集中：**v1 的固定测试集从一开始就是 100% 合成噪声**——训练阶段在线混音确实
用了真实 MUSAN 噪声，但拿来算 SI-SDR/STOI/PESQ 最终指标的那份 `data/testset/`，
噪声全部来自 `make_noise()`。这意味着 v1 报告的指标可能对真实世界噪声的代表性存疑，
这正是这一版存在的理由。

## 数据构成

| 组成 | 来源 | 体积 | 相对 v1 的变化 |
|---|---|---|---|
| 干净语音 | THCHS-30（openSLR SLR18） | ~6.4 GB | **不变**——DNS Challenge 没有中文语料，换了会让训练语音和中文 ASR 评测脱节 |
| 训练噪声 | DNS Challenge 5 `noise_fullband`：AudioSet + Freesound 各 1 个分片 | ~9 GB（估） | 替换 MUSAN noise 子集 |
| 训练 RIR | DNS Challenge 5 `impulse_responses` 分片 | ~5.9 GB | 替换 RIRS_NOISES |
| 冲激型噪声 | 本项目合成（跟 v1 完全一样） | 0 | **不变**——针对 `docs/FINDINGS.md` F-02 的专项压力测试，真实噪声库不保证覆盖 |
| **固定测试集** | 合成噪声分层（跟 v1 同构）+ **新增真实 DNS 噪声分层** | — | 两层并存，可直接对比 |

## 为什么是这个子集，不是完整 DNS Challenge 语料

DNS Challenge 5 完整语料（多语言干净语音 + 全部噪声/IR 分片）总计接近 **1 TB**
（干净语音各语种合计约 827 GB，噪声全部分片约 58 GB，IR 全部分片约 5.9 GB）。
本地磁盘装不下（C: 盘预算 <5GB，见 [`docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md)），
Colab 也没必要——这是一个刻意选小、但仍然真实、多来源（AudioSet 日常环境声 +
Freesound 标注音效）的子集：**2 个噪声分片 + 1 个 IR 分片，约 20 GB**，
在 Colab Pro（据反馈本地盘 240 多 GB）上轻松装得下。

## 数据来源与几个诚实的说明

- **下载 URL**：来自官方仓库 `microsoft/DNS-Challenge` 的
  `download-dns-challenge-5-noise-ir.sh`（2026-08 核对），Azure 公开 blob
  （`dnschallengepublic.blob.core.windows.net`），不需要鉴权，plain `wget` 即可。
- **分片体积未逐一实测**：本机磁盘装不下几 GB 的分片来提前验证下载是否可行
  （见上面的磁盘预算），notebook 里按官方文档"全部噪声分片共 ~39GB / 9 个分片"
  估算的均摊值做磁盘预检，实际大小以 Colab 上 `wget` 显示的为准。
- **分片解压后的内部目录结构未验证**：同样因为本机没法下载确认。`fetch_dns_blob()`
  的校验逻辑因此没有像 openSLR 那版一样断言一个具体子目录名，而是改成
  "解压后递归扫到的 wav 文件数量"——只要文件确实解出来了就能识别成功，不依赖
  对内部打包结构的假设是否准确。
- **混音脚本**：没有直接用 DNS Challenge 自带的 `noisyspeech_synthesizer_singleprocess.py`，
  继续用本项目自己的 `rtse.data.synth.mix_at_snr`/`apply_rir`/`make_rir`——
  这些函数已经跟项目的 dBFS 标定（`spec_ref()`/`magnitude_db()`）和 STFT/COLA
  管线对齐并测试过，引入第二套独立的混音/响度归一化实现只会制造新的不一致风险
  （类似 `docs/ISSUES.md` I-07 那次"写盘再读回来"污染指标的教训），
  DNS Challenge 真正有独特价值的是**数据本身**，不是它的胶水脚本。

这些"未验证"不是疏漏，是本机资源约束下的诚实记录——第一次在 Colab 上跑
`01_data_prep.ipynb` 就是真正的验证，脚本里对应位置都有防御性校验和清晰的失败信息。

## 固定测试集：新增的真实噪声分层

跟 v1 完全同构的合成噪声主表（SNR × 6 类噪声，T60=0.3）+ 混响扫描照抄不改，
保证能直接跟 v1 对比。新增一组：同样扫 6 档 SNR，噪声取自训练阶段划分出来的
`nz_test`（真实 DNS 噪声文件，训练从未见过），T60 固定 0.3。每条记录的
`index.json` 里带 `noise_source` 字段（`synthetic` / `real_dns`），
本地汇总时按这个字段分组，就能直接回答"这个模型在真实噪声上到底表现如何，
跟合成噪声测出来的数字差多少"。

## Drive 目录布局

```
MyDrive/Audio AI/RTSE/                    ← CODE_ROOT，v1/v2 共用（rtse-colab.zip）
├── rtse-colab.zip
├── manifest.json                          ← v1 的
├── checkpoints/ models/ testset/          ← v1 的
└── v2_dns_real_noise/                     ← DRIVE_ROOT（本版本专属，物理隔离）
    ├── manifest.json
    ├── archives/*.tar.bz2
    ├── checkpoints/<模型名>/
    ├── models/<模型名>.onnx
    └── testset/ + testset.zip
```

`rtse-colab.zip` 只需要上传一次，v1、v2 的 notebook 都从 `CODE_ROOT` 读它；
其余产物（数据、checkpoint、测试集）各版本互相隔离，不会覆盖。

## 三个 notebook

跟 v1 结构一一对应，`02_train.ipynb`/`03_export_eval.ipynb` 的训练/导出/评测代码
本身跟 v1 完全相同（模型不关心噪声是合成的还是真实录制的），差异只集中在
`01_data_prep.ipynb` 的数据下载部分和三个 notebook 共用的「配置」cell
（`CODE_ROOT`/`DRIVE_ROOT` 拆分）。

1. `01_data_prep.ipynb` —— 下载 THCHS-30 + DNS 噪声/IR 分片，合成含真实噪声分层的固定测试集
2. `02_train.ipynb` —— 训练三档模型（跟 v1 相同代码，数据源不同）
3. `03_export_eval.ipynb` —— 导出流式 ONNX + 评测（新增按 `noise_source` 分组的汇总）

## 尚未做的事（诚实记录）

- 还没有在 Colab 上实际跑过——本文档和三个 notebook 是**生成后做了语法/shell 引号
  静态校验**（`compile()` + 空格安全检查全部通过），但下载 URL 是否仍然有效、
  分片解压后是否真的能被 `scan()` 扫到，只有真正跑一次才能确认。
- v1、v2 两版模型目前还没有在同一份评测报告里并排比较过——这是后续要做的事，
  等 v2 训练完、testset 下载回本地之后再补。
