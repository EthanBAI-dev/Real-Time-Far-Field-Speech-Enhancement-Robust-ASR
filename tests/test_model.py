"""模型测试。核心是**因果性**与**流式一致性** —— 这两条不过，ONNX 导出就没有意义。"""

import numpy as np
import pytest
import torch

from rtse.models import PRESETS, build_model

torch.manual_seed(20260805)


@pytest.fixture(scope="module", params=sorted(PRESETS))
def model(request):
    m = build_model(request.param)
    m.eval()
    return m


def test_shapes_and_param_budget(model):
    spec = torch.randn(2, 2, 30, 257)
    with torch.no_grad():
        out = model(spec)
    assert out.shape == spec.shape
    assert torch.isfinite(out).all()

    n = model.count_params()
    name = [k for k, v in PRESETS.items() if v == model.cfg][0]
    budget = {"crn-nano": 200_000, "crn-lite": 1_000_000, "crn-large": 4_000_000}[name]
    assert n < budget, f"{name} 参数量 {n} 超出预算 {budget}"


def test_model_is_strictly_causal(model):
    """**最重要的一条测试。**

    改动第 t 帧之后的输入，第 t 帧及之前的输出必须**一个比特都不变**。
    只要有一处对称 padding 或双向 RNN，这条就会失败 ——
    而离线指标依然会很好看，所以不测的话根本发现不了。
    """
    spec = torch.randn(1, 2, 24, 257)
    t = 12
    with torch.no_grad():
        out_a = model(spec)
        perturbed = spec.clone()
        perturbed[:, :, t + 1 :] = torch.randn_like(perturbed[:, :, t + 1 :]) * 10
        out_b = model(perturbed)

    err = (out_a[:, :, : t + 1] - out_b[:, :, : t + 1]).abs().max().item()
    assert err < 1e-6, f"未来帧影响了当前输出（偏差 {err:.3e}）—— 模型不是因果的"


def test_streaming_matches_batch(model):
    """逐帧流式推理必须与整段推理数值一致。

    这是 ONNX 流式导出能够成立的前提：导出的是单帧步进图，
    如果 PyTorch 侧的流式路径就与整段对不上，导出后只会更糟。
    """
    spec = torch.randn(1, 2, 40, 257)
    with torch.no_grad():
        batch_out = model(spec)
        state = model.init_state(batch=1)
        frames = []
        for t in range(spec.shape[2]):
            y, state = model.forward_stream(spec[:, :, t : t + 1], state)
            frames.append(y)
        stream_out = torch.cat(frames, dim=2)

    err = (batch_out - stream_out).abs().max().item()
    assert err < 1e-5, f"流式与整段不一致（偏差 {err:.3e}）"


def test_state_shapes_are_stable(model):
    """流式状态的形状必须逐帧不变 —— ONNX 图的输入输出形状是固定的，
    状态一旦变形，导出的图就用不了。"""
    spec = torch.randn(1, 2, 5, 257)
    state = model.init_state(batch=1)
    shapes = [tuple(s.shape) for s in state]
    with torch.no_grad():
        for t in range(5):
            _, state = model.forward_stream(spec[:, :, t : t + 1], state)
            assert [tuple(s.shape) for s in state] == shapes


def test_mask_is_bounded(model):
    """极端输入下输出不得爆炸。CRM 的 tanh 限幅就是为这个服务的。"""
    for scale in [1e-6, 1.0, 1e4]:
        spec = torch.randn(1, 2, 10, 257) * scale
        with torch.no_grad():
            out = model(spec)
        assert torch.isfinite(out).all()
        ratio = (out.abs().max() / spec.abs().max().clamp_min(1e-12)).item()
        assert ratio < model.cfg.mask_bound * 3, f"输入尺度 {scale} 下输出放大了 {ratio:.1f} 倍"


def test_silence_in_silence_out(model):
    """全零输入必须给出全零输出（CRM 是乘性的，这是结构上的必然）。"""
    with torch.no_grad():
        out = model(torch.zeros(1, 2, 10, 257))
    assert out.abs().max().item() < 1e-6


def test_describe_reports_causal_receptive_field(model):
    d = model.describe()
    assert d["causal"] is True
    assert d["freq_dims"][0] == 257
    assert d["receptive_field_frames"] == len(model.encoder) * (model.cfg.kernel_time - 1) + 1
