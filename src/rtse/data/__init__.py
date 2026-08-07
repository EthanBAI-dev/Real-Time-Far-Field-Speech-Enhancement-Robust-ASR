"""数据层：合成、清单、Dataset。

``synth`` 在本地与 Colab 两侧共用 —— SNR 定义、混音方式、活跃段判定
必须两边**完全一致**，否则本地评测的"0 dB"和 Colab 训练的"0 dB"不是同一件事。
"""

from rtse.data.synth import (
    NOISE_KINDS,
    apply_rir,
    make_noise,
    make_rir,
    mix_at_snr,
    speech_active_mask,
)

__all__ = [
    "NOISE_KINDS",
    "make_noise",
    "make_rir",
    "mix_at_snr",
    "apply_rir",
    "speech_active_mask",
]
