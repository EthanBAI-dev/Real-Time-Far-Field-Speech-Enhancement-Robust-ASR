"""checkpoint 在 CPU / GPU 运行时之间搬来搬去必须不炸。

**由来**：Colab 上切换运行时是常态（GPU 配额用完会被降级到 CPU）。
在 CPU 会话里存下的 `last.pt`，拿到 GPU 会话续训时直接崩：

    RuntimeError: The source state dict is empty, possibly because it was
    saved from a disabled instance of GradScaler.

根因是 `GradScaler` 被禁用时 `state_dict()` 返回空字典，
而空字典喂给**启用**状态的 `load_state_dict()` 会抛异常。
`load()` 原来无条件搬这个字段，没考虑存档端与加载端的 AMP 开关可能不一致。
见 docs/ISSUES.md I-36。

## 为什么不能直接 `GradScaler(enabled=True)` 来测

本机（以及 CI）没有 CUDA，`torch.amp.GradScaler("cuda", enabled=True)` 会
**自动降级为禁用**并打一条 UserWarning。第一版测试正是这么写的，
结果四种存/读组合跑的全是"禁用↔禁用"这一条路径，
`load_state_dict({})` 根本不会抛错——**测试全绿，却一个字节的 bug 也没碰到**。

所以这里用一个**强制表现为启用**的替身 scaler：它的 `load_state_dict`
复刻 PyTorch 在启用状态下遇到空字典时的真实行为（抛 RuntimeError）。
这样在没有 GPU 的机器上也能真实复现那一格。
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from rtse.models import build_model
from rtse.train import Trainer, TrainConfig


class _EnabledScaler(torch.amp.GradScaler):
    """行为等同于「CUDA 可用时启用的 GradScaler」，可在无 GPU 机器上构造。

    只覆写与本测试相关的三个方法：``is_enabled``、``state_dict``、
    ``load_state_dict``。其中 ``load_state_dict`` 对空字典抛 RuntimeError，
    与上游 `torch/amp/grad_scaler.py` 的实际实现一致——
    这正是我们要复现的那一步。
    """

    def is_enabled(self) -> bool:  # noqa: D102
        return True

    def state_dict(self):  # noqa: D102
        return {"scale": 65536.0, "growth_tracker": 0}

    def load_state_dict(self, state_dict):  # noqa: D102
        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )


class _TinyDS(Dataset):
    """两条极短的假样本。这里要验的是 checkpoint 的存取，不是训练效果。"""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(idx)
        x = torch.randn(16000, generator=g)
        return x, x.clone()


def _trainer(tmp_path, use_amp: bool) -> Trainer:
    """造一个 Trainer 并把它的 AMP 开关**摆成指定状态**，模拟不同运行时。"""
    dl = DataLoader(_TinyDS(), batch_size=1)
    tr = Trainer(build_model("crn-nano"), dl, dl,
                 TrainConfig(model="crn-nano", epochs=1, out_dir=str(tmp_path)),
                 device="cpu")
    tr.use_amp = use_amp
    tr.scaler = (_EnabledScaler("cuda", enabled=False) if use_amp
                 else torch.amp.GradScaler("cuda", enabled=False))
    return tr


def test_the_failure_mode_is_actually_reproducible_here():
    """先证明**这套替身真的能复现那个错误**，否则下面的测试就是自欺欺人。

    第一版测试因为本机无 CUDA 而全程走"禁用"路径，全绿但什么都没测到。
    这条测试守住那个前提：空字典喂给"启用"的 scaler 必须抛 RuntimeError。
    """
    disabled = torch.amp.GradScaler("cuda", enabled=False)
    assert disabled.state_dict() == {}, "禁用的 scaler 应当序列化成空字典"

    with pytest.raises(RuntimeError, match="source state dict is empty"):
        _EnabledScaler("cuda", enabled=False).load_state_dict(disabled.state_dict())


@pytest.mark.parametrize(
    "saved_with_amp,loaded_with_amp",
    [(False, True), (True, False), (False, False), (True, True)],
    ids=["cpu存-gpu读", "gpu存-cpu读", "cpu存-cpu读", "gpu存-gpu读"],
)
def test_checkpoint_survives_amp_mismatch(tmp_path, saved_with_amp, loaded_with_amp):
    """四种存/读组合都必须能恢复，**不能因为 AMP 开关不一致就崩**。

    `cpu存-gpu读` 就是实际踩到的那一格：CPU 会话存下的 scaler 是空字典，
    修复前会在这里抛 RuntimeError。
    """
    src = _trainer(tmp_path, use_amp=saved_with_amp)
    src.epoch, src.step = 7, 123
    ckpt = src.save("last.pt")

    dst = _trainer(tmp_path, use_amp=loaded_with_amp)
    dst.load(ckpt)

    assert dst.epoch == 7, "续训进度必须恢复——这才是 checkpoint 的意义"
    assert dst.step == 123


def test_model_weights_are_intact_across_amp_mismatch(tmp_path):
    """跨 AMP 边界恢复后，**权重必须逐值相同**。

    只断言 epoch 恢复是不够的：真正要保证的是"换个运行时接着训"
    等价于"没换过"，那要求模型参数一个比特都不能变。
    """
    src = _trainer(tmp_path, use_amp=False)
    ckpt = src.save("last.pt")
    dst = _trainer(tmp_path, use_amp=True)
    dst.load(ckpt)

    for (k1, v1), (k2, v2) in zip(src.model.state_dict().items(),
                                  dst.model.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2), f"参数 {k1} 在跨 AMP 恢复后变了"
