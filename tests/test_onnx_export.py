"""ONNX 流式导出与集成测试。

**这是本项目工程含量最高的一环**：导出的不是整段图，而是单帧步进图，
状态（卷积缓存 + GRU 隐状态）必须显式外置成图的输入输出。
任何一处状态传递写错，模型都还是"能跑"的 —— 只是效果悄悄变差。
所以必须靠数值一致性来验证，不能靠"跑通了"。
"""

import numpy as np
import pytest
import torch

from rtse.models import build_model
from rtse.train.export import export_streaming_onnx, verify_onnx_streaming

MODELS = ["crn-nano", "crn-lite"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """导出一次，多条测试共用。导出较慢，不宜每条测试都重来。"""
    d = tmp_path_factory.mktemp("onnx")
    out = {}
    for name in MODELS:
        m = build_model(name).eval()
        # 随机初始化的模型输出接近零，数值一致性会被浮点噪声主导。
        # 稍微扰动权重，让输出有实际动态范围，一致性检验才有意义。
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(1.0).add_(torch.randn_like(p) * 0.05)
        out[name] = (m, export_streaming_onnx(m, d / f"{name}.onnx", verify=False))
    return out


@pytest.mark.parametrize("name", MODELS)
def test_export_produces_expected_graph_signature(exported, name):
    _, info = exported[name]
    # 输入 = 1 帧 + 5 个卷积 cache + 1 个 GRU 状态
    assert info["input_names"][0] == "frame"
    assert len(info["input_names"]) == 1 + 5 + 1
    assert len(info["output_names"]) == len(info["input_names"])
    assert info["state_shapes"]["cache_0"] == [1, 2, 1, 257]
    assert info["size_kb"] > 0


@pytest.mark.parametrize("name", MODELS)
def test_onnx_streaming_matches_pytorch_batch(exported, name):
    """**核心判据**：ONNX 逐帧推理必须与 PyTorch 整段推理一致。

    这一条同时证明了三件事：ONNX 导出正确、状态传递正确、模型确实是因果的。
    只测"ONNX vs PyTorch 流式"是不够的 —— 若 PyTorch 的流式实现本身有错，
    两边会一致地错。必须以**整段**推理作为基准。
    """
    m, info = exported[name]
    v = verify_onnx_streaming(m, info["path"], n_frames=50)
    assert v["state_shape_stable"], "状态形状逐帧变化，ONNX 图无法使用"
    assert v["relative_error"] < 1e-4, f"ONNX 流式与 PyTorch 整段相对偏差 {v['relative_error']:.3e}"
    assert v["passed"]


@pytest.mark.parametrize("name", MODELS)
def test_onnx_enhancer_plugs_into_pipeline(exported, name):
    """ONNX 模型必须能无差别地接进 Pipeline，与 DSP 方法共用全部下游代码。"""
    from rtse.runtime import OnnxEnhancer, Pipeline
    from rtse.vad import build_vad

    _, info = exported[name]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000) * 0.1

    pipe = Pipeline(enhancer=OnnxEnhancer(info["path"]), vad=build_vad("energy"))
    out, frames = pipe.process_signal(x)

    assert out.size == x.size
    assert np.all(np.isfinite(out))
    st = pipe.latency_stats()
    assert st.n_frames == len(frames)
    assert st.p99_ms < st.frame_budget_ms, f"p99 {st.p99_ms:.2f} ms 超出帧预算"
    # 等效增益必须给得出来，Web 界面靠它做 NN 与 DSP 的并排对比
    assert frames[50].gain_db is not None


@pytest.mark.parametrize("name", MODELS)
def test_onnx_enhancer_reset_is_clean(exported, name):
    """reset() 后重跑必须完全一致 —— Web 端热切换方法依赖这一点。"""
    from rtse.runtime import OnnxEnhancer

    _, info = exported[name]
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8000) * 0.1
    enh = OnnxEnhancer(info["path"])
    a = enh.process(x)
    b = enh.process(x)
    assert np.max(np.abs(a - b)) < 1e-9


def test_discover_skips_subdirectories(tmp_path):
    """模型发现只扫顶层，不能把 _untrained/ 之类的辅助目录也列进去。"""
    from rtse.runtime import discover_onnx_models

    (tmp_path / "good.onnx").write_bytes(b"x")
    sub = tmp_path / "_untrained"
    sub.mkdir()
    (sub / "bad.onnx").write_bytes(b"x")

    found = discover_onnx_models(tmp_path)
    assert set(found) == {"good"}
