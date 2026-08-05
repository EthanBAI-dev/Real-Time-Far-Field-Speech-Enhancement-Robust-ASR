"""增强器的统一接口。

**核心设计决定：离线处理由流式实现驱动。**

``process()`` 不是独立的整段实现，而是内部调用 ``StreamingSTFT`` → ``process_frame()``
→ ``StreamingISTFT`` 逐帧跑完。这样一来：

- 离线评测的数字**就是**实时部署时的数字，不存在"论文里好、上线就崩"；
- 不需要为"离线/流式一致性"额外写一套对照实现，一致性由结构保证；
- 新增算法只要实现一个 ``process_frame``，就自动同时拥有离线与实时两种能力。

代价是离线处理慢于向量化的整段实现。对本项目无所谓：
评测集只有几百条，而"指标可信"远比"评测快 3 秒"重要。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig, StreamingISTFT, StreamingSTFT, num_frames

__all__ = ["StreamingEnhancer", "PassThrough"]


class StreamingEnhancer(ABC):
    """逐帧语音增强器基类。

    子类只需实现 ``process_frame``（以及可选的 ``reset``）。
    """

    #: 在 Web 界面与评测报告中显示的名字
    name: str = "base"

    def __init__(self, cfg: STFTConfig = DEFAULT_CONFIG) -> None:
        self.cfg = cfg

    @abstractmethod
    def process_frame(self, spec: np.ndarray) -> np.ndarray:
        """输入一帧带噪复数谱 ``(n_freq,)``，返回同形状的增强复数谱。"""

    def reset(self) -> None:
        """清空内部状态。Web 端热切换方法时必须调用，否则会串上一段的噪声估计。"""

    @property
    def last_gain(self) -> np.ndarray | None:
        """最近一帧施加的实数增益（可视化用）。没有则返回 None。"""
        return None

    def process(self, x: np.ndarray) -> np.ndarray:
        """整段处理。内部逐帧跑，因此结果与实时管线逐样本相同。"""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        n_fr = num_frames(x.size, self.cfg)
        if n_fr == 0:
            return np.zeros(0)

        feed = np.zeros(n_fr * self.cfg.hop)
        feed[: x.size] = x

        self.reset()
        sa, sy = StreamingSTFT(self.cfg), StreamingISTFT(self.cfg)
        out = np.concatenate(
            [
                sy.push(self.process_frame(sa.push(feed[i * self.cfg.hop : (i + 1) * self.cfg.hop])))
                for i in range(n_fr)
            ]
        )
        # 丢弃左侧补零区，对齐回原始时间轴
        return out[self.cfg.pad : self.cfg.pad + x.size]


class PassThrough(StreamingEnhancer):
    """直通。作为 ``noisy`` 基线，也用来验证"增强器外壳本身不引入误差"。"""

    name = "passthrough"

    def process_frame(self, spec: np.ndarray) -> np.ndarray:
        return spec
