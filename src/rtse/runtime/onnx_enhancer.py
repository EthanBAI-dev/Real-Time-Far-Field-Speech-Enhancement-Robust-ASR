"""把流式 ONNX 模型包装成 ``StreamingEnhancer``。

这个适配器是整个架构的收口：包好之后，神经模型与谱减、维纳、MMSE-LSA
在**所有**下游代码里完全等价 —— 同一条 ``Pipeline``、同一套延迟计量、
同一个 Web 演示、同一份评测器，一行特判都不需要。

好处不只是代码干净。它保证了 "NN vs DSP" 的对比是**公平的**：
两者走的是同一条流式路径、同样的 STFT、同样的计时点。
如果神经模型走一条特殊的批处理快车道，测出来的 RTF 就没有可比性。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig
from rtse.dsp.base import StreamingEnhancer

__all__ = ["OnnxEnhancer", "discover_onnx_models"]


class OnnxEnhancer(StreamingEnhancer):
    """流式 ONNX 增强器。

    Args:
        path: ``.onnx`` 文件路径。
        threads: onnxruntime 的 intra-op 线程数。**默认 1** ——
            RTF 与延迟指标必须在单线程下测才有部署参考价值。
    """

    def __init__(
        self, path: str | Path, cfg: STFTConfig = DEFAULT_CONFIG, threads: int = 1
    ) -> None:
        super().__init__(cfg)
        from rtse.train.export import OnnxStreamingModel

        self.path = Path(path)
        self.name = self.path.stem
        self._model = OnnxStreamingModel(self.path, intra_threads=threads)
        self._last_gain: np.ndarray | None = None

    def reset(self) -> None:
        self._model.reset()
        self._last_gain = None

    @property
    def last_gain(self) -> np.ndarray | None:
        """等效实数增益 = |输出| / |输入|。

        神经模型施加的是复数掩码，没有"增益"这个直接概念。
        但把等效幅度增益算出来，就能在 Web 界面上与 DSP 方法**并排比较**
        它们各自在哪些频段动了手 —— 这是理解模型行为最直观的一张图。
        """
        return self._last_gain

    def process_frame(self, spec: np.ndarray) -> np.ndarray:
        out = self._model.process_frame(spec)
        self._last_gain = np.abs(out) / (np.abs(spec) + 1e-12)
        return out


def discover_onnx_models(models_dir: str | Path | None = None) -> dict[str, Path]:
    """扫描 models/ 下的 .onnx 文件。Web 服务与评测器据此列出可用的神经模型。"""
    from rtse.paths import MODELS_DIR

    root = Path(models_dir) if models_dir else MODELS_DIR
    if not root.exists():
        return {}
    # 排除 dnsmos 等辅助模型目录，只取顶层的增强模型
    return {p.stem: p for p in sorted(root.glob("*.onnx"))}
