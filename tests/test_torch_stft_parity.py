"""torch 版 STFT/iSTFT 与 numpy 版的一致性。

**为什么这条测试非做不可**：训练在 torch 里做 STFT，推理在 numpy/ONNX 里做 STFT。
两者只要有一点不一致（center padding、窗函数、归一化、补零方式任一处），
模型见到的输入分布就与部署时不同 —— 表现是"训练验证集上很好，一上线就掉点"，
而且几乎无从定位，因为两边单独看都"没问题"。
"""

import numpy as np
import pytest
import torch

from rtse.audio.stft import DEFAULT_CONFIG, istft, stft
from rtse.data.dataset import istft_torch, stft_torch

RNG = np.random.default_rng(20260805)


@pytest.mark.parametrize("n", [4000, 16000, 12345])
def test_stft_torch_matches_numpy(n):
    x = RNG.standard_normal(n)
    ref = stft(x)  # (T, F) complex
    got = stft_torch(torch.from_numpy(x).float().unsqueeze(0))  # (1, 2, T, F)

    # 帧数必须**精确相等**。早期这里放宽到 ±1，掩盖了 torch 侧少算一帧的 bug ——
    # STFT 和 iSTFT 各自对拍都能过，只有完整往返才暴露尾部丢失（I-14）。
    assert ref.shape[0] == got.shape[2], f"帧数不一致 {ref.shape[0]} vs {got.shape[2]}"

    t = ref.shape[0]
    got_c = got[0, 0, :t].numpy() + 1j * got[0, 1, :t].numpy()
    # float32 精度下，幅度约 100 量级的谱，1e-2 是合理的容差
    err = np.max(np.abs(got_c - ref[:t]))
    rel = err / (np.max(np.abs(ref[:t])) + 1e-12)
    assert rel < 1e-4, f"torch/numpy STFT 相对偏差 {rel:.3e}"


def test_istft_torch_matches_numpy():
    x = RNG.standard_normal(8000)
    spec = stft(x)
    ref = istft(spec, length=x.size)

    spec_t = torch.stack(
        [torch.from_numpy(spec.real).float(), torch.from_numpy(spec.imag).float()]
    ).unsqueeze(0)
    got = istft_torch(spec_t, length=x.size)[0].numpy()

    rel = np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12)
    assert rel < 1e-4, f"torch/numpy iSTFT 相对偏差 {rel:.3e}"


def test_torch_roundtrip_reconstructs():
    """torch 侧自身的 STFT→iSTFT 也必须是完美重构。"""
    x = torch.from_numpy(RNG.standard_normal(8000)).float().unsqueeze(0)
    y = istft_torch(stft_torch(x), length=x.shape[-1])
    rel = (y - x).abs().max().item() / x.abs().max().item()
    assert rel < 1e-4, f"torch 往返重构相对误差 {rel:.3e}"


def test_batched_stft_is_consistent():
    """批处理与逐条处理必须一致（BatchNorm 之外不该有任何跨样本耦合）。"""
    xs = torch.from_numpy(RNG.standard_normal((4, 8000))).float()
    batched = stft_torch(xs)
    for i in range(4):
        single = stft_torch(xs[i : i + 1])
        assert torch.allclose(batched[i : i + 1], single, atol=1e-4)


def test_model_pipeline_end_to_end():
    """完整训练前向：波形 → STFT → 模型 → iSTFT → 波形，形状与数值都要正常。"""
    from rtse.models import build_model

    m = build_model("crn-nano").eval()
    wav = torch.from_numpy(RNG.standard_normal((2, 16000))).float() * 0.3
    with torch.no_grad():
        spec = stft_torch(wav)
        out_spec = m(spec)
        out = istft_torch(out_spec, length=wav.shape[-1])
    assert out.shape == wav.shape
    assert torch.isfinite(out).all()
