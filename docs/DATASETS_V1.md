# V1 数据集与证据链

## 1. 任务边界

V1 是**单通道实时加性噪声抑制**。RIR 用来构造混响环境，但训练与受控评测的
目标都是“带混响、无加性噪声”的语音，因此不把混响残留算作去噪错误，也不宣称去混响。

## 2. 数据职责

| 名称 | 来源 | 规模 | 是否有有效无噪参考 | 允许的指标 |
|---|---|---:|---|---|
| 训练语音 | DNS Challenge 5 `read_speech` 首个可独立部分解压切片 | 约19小时 | — | 训练 |
| 训练噪声 | DNS5 AudioSet×2 + Freesound×1 | 约14.2GB压缩 | — | 训练 |
| RIR | DNS5真实RIR + 镜像源法合成RIR | 60248条真实RIR | — | 训练/分层 |
| `dns_objective` | DNS5留出说话人、噪声和RIR | 冒烟81 / 正式405 | 是 | SI-SDR、STOI、ESTOI、PESQ、DNSMOS |
| `aishell_controlled` | AISHELL-1 test + 同一受控退化矩阵 | 冒烟81 / 正式405 | 是，且有文本 | 上述质量指标 + CER |
| `wenetspeech_real` | WenetSpeech `test_meeting` 原始音频 | 冒烟30 / 正式300 | 否，只有人工文本 | 输入/增强 CER |

AISHELL-1 官方来源是 OpenSLR SLR33；notebook 使用 test-only parquet 镜像第一片
（约398MB），避免为了几百条评测语音下载15GB全量训练包。WenetSpeech 使用公开
parquet镜像。镜像只改变打包方式，不改变音频和人工转写的职责。

## 3. 训练/验证隔离

- DNS语音按文件名中的 `reader_XXXXX` 说话人ID划分，不能按文件随机切分。
- 噪声与真实RIR先固定随机划分；训练只用train部分，三套测试只用test部分。
- AISHELL与WenetSpeech从不参与增强模型训练。
- 在线训练混音与固定测试混音使用不同随机种子；测试音频预生成后不再变化。

## 4. 受控矩阵

噪声/RIR矩阵为：

`SNR 5档 × 噪声平稳性2类 × RIR来源2类 × RT60桶4档 = 80格`

- SNR：−5 / 0 / 5 / 10 / 15 dB
- 噪声：stationary / nonstationary
- RIR来源：synth / real
- RT60桶：0.2 / 0.4 / 0.6 / 0.8秒
- 另加1个 `clean→clean` identity格，专门检查无害性

真实RIR不是随便抽取：先用Schroeder反向积分实测RT60，再放入对应桶，只与同档
合成RIR比较。0.2/0.4/0.6/0.8秒桶范围分别为[0.10,0.30)、[0.30,0.50)、
[0.50,0.70)、[0.70,1.00)秒。更长的真实RIR留作未来压力测试，不进入主对比。

## 5. `index.json` 能力声明

每套数据自己声明评测前提：

```json
{
  "reference_is_clean": true,
  "cer_upper_is_meaningful": true,
  "strata": ["snr", "noise_kind", "rir_kind", "rt60_bucket"]
}
```

`rtse-eval` 不再根据目录名猜测：`reference_is_clean=false` 时自动跳过有参考指标；
`cer_upper_is_meaningful=false` 时不读取 `clean`，也不输出虚假的“clean上界”。

## 6. 运行规模与结论权限

| 模式 | 受控集 | 真实会议集 | 训练 | 可以下什么结论 |
|---|---:|---:|---|---|
| `SMOKE_RUN=True` | 81条/套 | 30条 | Nano，2000样本×3 epoch | 只证明流水线可运行 |
| `SMOKE_RUN=False` | 405条/套 | 300条 | Nano+Lite，20000样本×60 epoch | 可形成正式主表，但需置信区间 |

冒烟结果不得写进简历或与公开基线比较。正式结果至少报告样本数、逐条件均值、
bootstrap 95%置信区间和失败案例。CER先用faster-whisper `small`全方法筛选，再用
`medium`复核 `none`、最佳DSP和最佳CRN，避免结论只依赖一个偏弱ASR。
