"""ONNX 流式导出 —— 本项目工程含量最高、也最容易做错的一环。

**导出的不是整段图，而是单帧步进图。**

ONNX 图是无状态的：它没有"上一帧留下的东西"这个概念。而流式推理必然有状态
（卷积的时间缓存、GRU 的隐状态）。所以唯一的办法是把状态**显式外置**成
图的输入和输出，由调用方在帧与帧之间自己传递::

    (当前帧, cache_0..4, gru_h) → (增强帧, new_cache_0..4, new_gru_h)

导出之后**必须**做三层校验，任何一层不过，后面所有实时指标都不可信：

1. **ONNX 与 PyTorch 流式一致**：同样逐帧喂，两边输出要一样；
2. **ONNX 流式与 PyTorch 整段一致**：这才是"流式没有偷看未来帧"的最终判据；
3. **状态形状逐帧不变**：ONNX 图的输入形状是固定的，状态一旦变形就没法用。

第 2 条是核心。第 1 条即使通过，如果 PyTorch 的流式实现本身就错了
（比如某层偷偷用了对称 padding），两边会"一致地错"。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rtse.models.crn import CRNLite

__all__ = ["export_streaming_onnx", "verify_onnx_streaming", "OnnxStreamingModel"]


class _StreamWrapper(torch.nn.Module):
    """把 ``forward_stream`` 的 list 状态摊平成位置参数，供 ONNX 导出。

    ONNX 不接受 list[Tensor] 作为图的输入/输出，必须摊成扁平的张量列表。
    """

    def __init__(self, model: CRNLite) -> None:
        super().__init__()
        self.model = model
        self.n_state = len(model.encoder) + 1

    def forward(self, frame: torch.Tensor, *state: torch.Tensor):
        out, new_state = self.model.forward_stream(frame, list(state))
        return (out, *new_state)


def export_streaming_onnx(
    model: CRNLite, path: str | Path, opset: int = 17, verify: bool = True
) -> dict:
    """导出单帧流式 ONNX 图。

    Returns:
        含参数量、文件大小、状态张量形状、校验结果的字典（直接写进评测报告）。
    """
    model = model.eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _StreamWrapper(model).eval()
    frame = torch.zeros(1, 2, 1, model.cfg.n_freq)
    state = model.init_state(batch=1, device=torch.device("cpu"))

    state_names = [f"cache_{i}" for i in range(len(model.encoder))] + ["gru_h"]
    input_names = ["frame", *state_names]
    output_names = ["out_frame", *[f"{n}_out" for n in state_names]]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (frame, *state),
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            # 刻意**不设 dynamic_axes**：流式图的形状是完全固定的
            # （batch=1、1 帧、257 频点）。固定形状让 onnxruntime 能做更激进的
            # 内存规划与算子融合，实测比动态形状快一截 —— 而我们并不需要动态性。
            dynamo=False,
        )

    info = {
        "path": str(path),
        "size_kb": round(path.stat().st_size / 1024, 1),
        "params": model.count_params(),
        "opset": opset,
        "input_names": input_names,
        "output_names": output_names,
        "state_shapes": {n: list(s.shape) for n, s in zip(state_names, state)},
        **model.describe(),
    }
    if verify:
        info["verification"] = verify_onnx_streaming(model, path)
    return info


def verify_onnx_streaming(model: CRNLite, path: str | Path, n_frames: int = 60) -> dict:
    """三层一致性校验。返回各项误差；判定标准写在返回值里，便于直接进报告。"""
    import onnxruntime as ort

    torch.manual_seed(0)
    spec = torch.randn(1, 2, n_frames, model.cfg.n_freq)

    # 基准 1：PyTorch 整段
    with torch.no_grad():
        batch_out = model(spec)

    # 基准 2：PyTorch 流式
    with torch.no_grad():
        st = model.init_state(1, device=torch.device("cpu"))
        pt_stream = []
        for t in range(n_frames):
            y, st = model.forward_stream(spec[:, :, t : t + 1], st)
            pt_stream.append(y)
        pt_stream = torch.cat(pt_stream, dim=2)

    # 被测：ONNX 流式
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    onnx_state = [s.numpy().astype(np.float32) for s in model.init_state(1, torch.device("cpu"))]
    shape_stable = True
    init_shapes = [s.shape for s in onnx_state]
    onnx_frames = []
    for t in range(n_frames):
        feeds = {in_names[0]: spec[:, :, t : t + 1].numpy().astype(np.float32)}
        feeds.update({n: s for n, s in zip(in_names[1:], onnx_state)})
        res = sess.run(out_names, feeds)
        onnx_frames.append(res[0])
        onnx_state = res[1:]
        if [s.shape for s in onnx_state] != init_shapes:
            shape_stable = False
    onnx_out = np.concatenate(onnx_frames, axis=2)

    err_onnx_vs_pt_stream = float(np.max(np.abs(onnx_out - pt_stream.numpy())))
    err_onnx_vs_pt_batch = float(np.max(np.abs(onnx_out - batch_out.numpy())))
    err_pt_stream_vs_batch = float(np.max(np.abs(pt_stream.numpy() - batch_out.numpy())))
    ref_scale = float(np.max(np.abs(batch_out.numpy()))) + 1e-12

    # 阈值按**相对**误差定。float32 累积 60 帧的误差与信号尺度成正比，
    # 用绝对阈值会在不同幅度的测试信号上给出不同结论。
    tol = 1e-4
    return {
        "n_frames": n_frames,
        "onnx_vs_pytorch_streaming": err_onnx_vs_pt_stream,
        "onnx_vs_pytorch_batch": err_onnx_vs_pt_batch,
        "pytorch_streaming_vs_batch": err_pt_stream_vs_batch,
        "relative_error": err_onnx_vs_pt_batch / ref_scale,
        "state_shape_stable": shape_stable,
        "tolerance": tol,
        "passed": bool(
            err_onnx_vs_pt_batch / ref_scale < tol
            and err_pt_stream_vs_batch / ref_scale < tol
            and shape_stable
        ),
    }


class OnnxStreamingModel:
    """本地推理端：加载流式 ONNX 图，提供与 ``StreamingEnhancer`` 一致的逐帧接口。

    这样神经模型就能直接插进现有的 ``Pipeline``，
    与 DSP 方法共用同一套延迟计量、Web 演示和评测代码 —— 不需要任何特判分支。
    """

    def __init__(self, path: str | Path, intra_threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # 单线程是**刻意**的：RTF 指标必须在单线程下测才有部署参考价值。
        # 多线程能让数字好看，但掩盖了真实的算力需求，也无法反映
        # 多路并发时的实际表现。
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self._init_state = [
            np.zeros([d if isinstance(d, int) else 1 for d in i.shape], dtype=np.float32)
            for i in self.sess.get_inputs()[1:]
        ]
        self.reset()

    def reset(self) -> None:
        self.state = [s.copy() for s in self._init_state]

    def process_frame(self, spec: np.ndarray) -> np.ndarray:
        """输入 ``(n_freq,)`` 复数谱，返回同形状的增强复数谱。"""
        frame = np.stack([spec.real, spec.imag]).astype(np.float32).reshape(1, 2, 1, -1)
        feeds = {self.in_names[0]: frame}
        feeds.update({n: s for n, s in zip(self.in_names[1:], self.state)})
        res = self.sess.run(self.out_names, feeds)
        self.state = res[1:]
        out = res[0].reshape(2, -1)
        return out[0].astype(np.float64) + 1j * out[1].astype(np.float64)
