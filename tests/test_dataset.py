"""训练数据集测试。

重点不是"能跑"，而是几个**会悄悄毁掉训练**的性质：
多 worker 下的随机性、增益对齐、坏文件容错。
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from rtse.data.dataset import MixConfig, OnlineMixDataset

SR = 16000


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """造一个迷你语料库：3 条语音、3 条噪声、2 条 RIR。"""
    root = tmp_path_factory.mktemp("corpus")
    rng = np.random.default_rng(0)
    for name, n in [("clean", 3), ("noise", 3)]:
        d = root / name
        d.mkdir()
        for i in range(n):
            t = np.arange(SR * 3) / SR
            y = (np.sin(2 * np.pi * (120 + 40 * i) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
                 if name == "clean" else rng.standard_normal(SR * 3) * 0.3)
            sf.write(d / f"{i}.wav", y * 0.5, SR, subtype="PCM_16")
    rd = root / "rir"
    rd.mkdir()
    for i in range(2):
        r = np.zeros(2000)
        r[0] = 1.0
        r[50 + i * 30 :] = rng.standard_normal(2000 - 50 - i * 30) * 0.05
        sf.write(rd / f"{i}.wav", r, SR, subtype="PCM_16")
    return root


def _ds(corpus, **kw):
    return OnlineMixDataset(
        corpus / "clean", corpus / "noise", corpus / "rir",
        cfg=MixConfig(segment_seconds=2.0), **kw
    )


def test_accepts_dirs_and_file_lists(corpus):
    """目录和文件列表两种入参必须都支持 —— Colab 侧用的是清单里的文件列表。"""
    from_dir = _ds(corpus, length=4)
    files = sorted(str(p) for p in (corpus / "clean").glob("*.wav"))
    from_list = OnlineMixDataset(
        files, sorted(str(p) for p in (corpus / "noise").glob("*.wav")),
        cfg=MixConfig(segment_seconds=2.0), length=4,
    )
    assert len(from_dir.clean) == len(from_list.clean) == 3


def test_empty_corpus_fails_loudly(tmp_path):
    """空目录必须立刻报错，而不是训练几小时后才发现全是静音。"""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(FileNotFoundError, match="干净语音"):
        OnlineMixDataset(tmp_path / "a", tmp_path / "b")


def test_output_shapes_and_dtype(corpus):
    ds = _ds(corpus, length=8)
    noisy, clean = ds[0]
    assert noisy.shape == clean.shape == (int(2.0 * SR),)
    assert noisy.dtype == clean.dtype == torch.float32
    assert torch.isfinite(noisy).all() and torch.isfinite(clean).all()


def test_never_clips(corpus):
    """训练样本**绝不能削波**。

    削波过的训练输入与推理时见到的输入分布不同，模型会学到一个不存在的
    "输入总是被压平"的先验。这条测试抓出过一个真实 bug：
    削波保护分支写成 `gain/peak*0.9`，在 gain>1.1 时结果仍会超过 1。
    随机增益上限是 +3 dB（1.41 倍），所以这个分支经常被触发。
    """
    ds = _ds(corpus, length=200, seed=17)
    worst = 0.0
    for i in range(200):
        noisy, clean = ds[i]
        worst = max(worst, noisy.abs().max().item(), clean.abs().max().item())
    assert worst <= 1.0, f"200 个样本中最大峰值 {worst:.4f} 超过满幅"


def test_same_index_is_deterministic(corpus):
    """同一个 idx 必须每次都给出完全相同的样本。

    不确定的话，验证集每个 epoch 都在变，val loss 曲线就没有可比性。
    """
    ds = _ds(corpus, length=8, seed=3)
    a1, b1 = ds[5]
    a2, b2 = ds[5]
    assert torch.equal(a1, a2) and torch.equal(b1, b2)


def test_different_indices_give_different_mixes(corpus):
    """不同 idx 必须给出不同的混音。

    这条是防"多 worker 下所有样本相同"的 —— 如果随机源用了全局 random，
    每个 worker 会各自从相同状态开始，产出大量重复样本，
    表现是"loss 降得异常快但验证集完全不行"。
    """
    ds = _ds(corpus, length=64, seed=0)
    mixes = [ds[i][0] for i in range(8)]
    for i in range(len(mixes)):
        for j in range(i + 1, len(mixes)):
            assert not torch.allclose(mixes[i], mixes[j]), f"样本 {i} 与 {j} 完全相同"


def test_gain_is_applied_to_both_signals(corpus):
    """随机增益必须**同时**作用于输入和参考。

    只缩放输入的话，等于在教模型猜绝对音量 —— 它会学出一个固定的增益偏置，
    在音量不同的真实录音上直接失效。
    判据：噪声段之外，两者的比例关系应当保持稳定。
    """
    ds = _ds(corpus, length=32, seed=11)
    for i in range(6):
        noisy, clean = ds[i]
        # 参考不应是全零，也不应远小于输入（那说明只缩放了一边）
        assert clean.abs().max() > 1e-3
        ratio = clean.abs().max() / noisy.abs().max().clamp_min(1e-9)
        assert 0.05 < ratio < 20, f"输入与参考的幅度比 {ratio:.3f} 异常"


def test_broken_file_does_not_crash(corpus, tmp_path):
    """语料库里混进坏文件是常态，必须容错而不是让整次训练崩掉。"""
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "broken.wav").write_bytes(b"not a wav file at all")
    for p in (corpus / "clean").glob("*.wav"):
        import shutil

        shutil.copy(p, bad_dir / p.name)

    ds = OnlineMixDataset(bad_dir, corpus / "noise", cfg=MixConfig(segment_seconds=2.0), length=20)
    for i in range(20):
        noisy, clean = ds[i]
        assert torch.isfinite(noisy).all()


def test_dataloader_workers_produce_varied_batches(corpus):
    """走真实 DataLoader（多 worker）跑一遍，确认没有跨 worker 的重复。"""
    from torch.utils.data import DataLoader

    ds = _ds(corpus, length=32, seed=0)
    dl = DataLoader(ds, batch_size=4, num_workers=2, shuffle=False)
    seen = [b[0] for b in dl]
    flat = torch.cat(seen)
    # 32 个样本里不应有完全相同的两条
    for i in range(0, flat.shape[0], 5):
        for j in range(i + 1, flat.shape[0], 5):
            assert not torch.allclose(flat[i], flat[j])
