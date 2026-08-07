"""神经网络模型。本地与 Colab 共用同一份定义。"""

from rtse.models.crn import PRESETS, CRNConfig, CRNLite, apply_crm, build_model, compress

__all__ = ["CRNLite", "CRNConfig", "PRESETS", "build_model", "compress", "apply_crm"]
