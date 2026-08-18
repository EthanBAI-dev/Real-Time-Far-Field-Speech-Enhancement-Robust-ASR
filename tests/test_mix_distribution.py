"""训练混音分布的覆盖性测试。

**由来**：模型在有噪输入上工作正常，却对**干净输入**下重手——
把干净参考喂进去，SI-SDR 从 128 dB 掉到 7.27 dB，
下游 CER 在高 SNR 段因此大幅恶化（docs/FINDINGS.md F-11）。

根因不是"数据不够多"，而是**训练分布缺了某一端**：SNR 上限只到 20 dB、
恒等样本只占 2%，模型从没见过"已经干净、不需要处理"的情形，
自然学不会此时放手。

这类缺陷**在训练和验证指标上完全看不出来**（验证集同分布，同样没有干净样本），
只有拿到分布外的真实输入才暴露。所以必须直接对**分布本身**做断言。
"""

import numpy as np
import pytest

from rtse.data.dataset import MixConfig, OnlineMixDataset
from rtse.metrics.intrusive import si_sdr


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    """造一批语音/噪声文件，跑一遍数据集，收集每个样本的实际输入 SI-SDR。"""
    import soundfile as sf

    d = tmp_path_factory.mktemp("mix")
    rng = np.random.default_rng(0)
    sp, nz = [], []
    for i in range(6):
        t = np.arange(16000 * 5) / 16000
        # 类语音：基频 + 谐波 + 包络，比纯噪声更接近真实语音的统计
        f0 = 110 + 20 * i
        x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3, 4))
        x *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
        p = d / f"sp{i}.wav"
        sf.write(p, (x / np.max(np.abs(x)) * 0.7), 16000)
        sp.append(str(p))
        q = d / f"nz{i}.wav"
        sf.write(q, rng.standard_normal(16000 * 5) * 0.1, 16000)
        nz.append(str(q))

    ds = OnlineMixDataset(sp, nz, [], cfg=MixConfig(), length=400, seed=0)
    out = []
    for i in range(len(ds)):
        noisy, target = ds[i]
        out.append(si_sdr(target.numpy(), noisy.numpy()))
    return np.asarray(out)


def test_distribution_reaches_near_clean_inputs(samples):
    """**必须存在接近无噪的训练样本**——这是 F-11 的直接回归测试。

    修复前 SNR 上限 20 dB、恒等样本 2%，这一条会失败。
    """
    near_clean = (samples > 30.0).mean()
    assert near_clean > 0.10, (
        f"只有 {near_clean:.1%} 的训练样本输入 SI-SDR > 30 dB。"
        f"模型见不到'已经干净'的情形就学不会放手，会对干净输入过抑制（F-11）"
    )


def test_distribution_still_covers_the_hard_low_snr_region(samples):
    """抬高上限**不能**把分布整体推向高 SNR。

    降噪真正要解决的是低 SNR，那一段的训练密度不能被稀释——
    否则修好了无害性、却换来主要场景变差。
    """
    hard = (samples < 5.0).mean()
    assert hard > 0.20, f"低 SNR(<5 dB) 样本只占 {hard:.1%}，主场景训练密度被稀释了"


def test_identity_examples_exist(samples):
    """有一批**完全不加噪**的恒等样本（target == input）。

    连续的高 SNR 尾巴和显式恒等样本作用不同：前者教"轻处理"，
    后者教"完全不处理"。两者都要有。
    """
    identity = (samples > 60.0).mean()
    assert identity > 0.03, (
        f"恒等样本只占 {identity:.1%}。noise_prob 应当留出足够比例的"
        f"不加噪样本（见 MixConfig.noise_prob 的说明）"
    )
