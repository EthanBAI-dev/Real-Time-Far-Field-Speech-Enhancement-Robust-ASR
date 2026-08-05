"""指标注册表：声明每个指标的方向、取值范围和当前可用性。

存在的理由：本项目的指标不是"全都能算"的 —— PESQ 在 Windows 上装不上（I-05），
DNSMOS 需要先下载 ONNX 模型，CER 需要转写文本。评测器必须能在缺指标时
**降级而不是崩溃**，并且在报告里如实标注哪一列是 n/a、为什么。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

__all__ = ["MetricSpec", "MetricResult", "METRIC_REGISTRY", "available_metrics"]

Direction = Literal["higher_is_better", "lower_is_better"]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    direction: Direction
    unit: str = ""
    value_range: tuple[float, float] | None = None
    needs_reference: bool = True
    probe: Callable[[], tuple[bool, str]] | None = None
    """返回 (是否可用, 不可用原因)。原因会原样写进评测报告。"""


@dataclass
class MetricResult:
    """单个样本上的一组指标值。缺失的指标为 None，报告渲染时显示 n/a。"""

    values: dict[str, float | None] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def set(self, key: str, value: float | None, note: str = "") -> None:
        self.values[key] = value
        if note:
            self.notes[key] = note


def _probe_pystoi() -> tuple[bool, str]:
    try:
        import pystoi  # noqa: F401
    except ImportError as exc:
        return False, f"pystoi 未安装: {exc}"
    return True, ""


def _probe_pesq() -> tuple[bool, str]:
    try:
        import pesq  # noqa: F401
    except ImportError:
        return False, "pesq 无 Windows wheel，需 MSVC 编译（见 ISSUES.md I-05）；改由 Colab 侧计算"
    return True, ""


def _probe_dnsmos() -> tuple[bool, str]:
    from pathlib import Path

    from rtse.paths import MODELS_DIR

    p = Path(MODELS_DIR) / "dnsmos" / "sig_bak_ovr.onnx"
    if not p.exists():
        return False, f"DNSMOS 模型未下载: {p}（在 Colab 侧获取后放到此路径）"
    return True, ""


METRIC_REGISTRY: dict[str, MetricSpec] = {
    m.key: m
    for m in [
        MetricSpec("si_sdr", "SI-SDR", "higher_is_better", "dB"),
        MetricSpec("sdr", "SDR", "higher_is_better", "dB"),
        MetricSpec("seg_snr", "SegSNR", "higher_is_better", "dB"),
        MetricSpec("stoi", "STOI", "higher_is_better", "", (0, 1), probe=_probe_pystoi),
        MetricSpec("estoi", "ESTOI", "higher_is_better", "", (0, 1), probe=_probe_pystoi),
        MetricSpec("pesq", "PESQ-WB", "higher_is_better", "", (-0.5, 4.5), probe=_probe_pesq),
        MetricSpec(
            "dnsmos_ovrl", "DNSMOS OVRL", "higher_is_better", "", (1, 5),
            needs_reference=False, probe=_probe_dnsmos,
        ),
        MetricSpec(
            "dnsmos_sig", "DNSMOS SIG", "higher_is_better", "", (1, 5),
            needs_reference=False, probe=_probe_dnsmos,
        ),
        MetricSpec(
            "dnsmos_bak", "DNSMOS BAK", "higher_is_better", "", (1, 5),
            needs_reference=False, probe=_probe_dnsmos,
        ),
        MetricSpec("cer", "CER", "lower_is_better", "%", needs_reference=False),
        MetricSpec("wer", "WER", "lower_is_better", "%", needs_reference=False),
    ]
}


def available_metrics() -> dict[str, tuple[bool, str]]:
    """探测所有指标的当前可用性。``rtse-doctor`` 与评测器都调用它。"""
    out: dict[str, tuple[bool, str]] = {}
    for key, spec in METRIC_REGISTRY.items():
        out[key] = spec.probe() if spec.probe else (True, "")
    return out
