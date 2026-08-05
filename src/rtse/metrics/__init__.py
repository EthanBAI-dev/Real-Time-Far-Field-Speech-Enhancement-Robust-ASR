"""客观指标。

分两类：
- **有参考**：SI-SDR、SDR、分段 SNR、STOI/ESTOI、PESQ —— 需要干净参考信号，
  只能在合成数据上算。
- **无参考**：DNSMOS —— 真实录音也能算，是"实时麦克风演示"页面唯一能显示的质量指标。

PESQ 在本机（Windows，无 MSVC）装不上，见 docs/ISSUES.md I-05。
这里做成"能导入就用、不能就返回 None"，评测表里标 n/a，而不是让整个评测崩掉。
"""

from rtse.metrics.intrusive import estoi, seg_snr, si_sdr, stoi
from rtse.metrics.registry import METRIC_REGISTRY, MetricResult, available_metrics

__all__ = [
    "si_sdr",
    "seg_snr",
    "stoi",
    "estoi",
    "METRIC_REGISTRY",
    "MetricResult",
    "available_metrics",
]
