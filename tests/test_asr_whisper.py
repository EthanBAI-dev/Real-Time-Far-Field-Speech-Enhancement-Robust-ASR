"""faster-whisper 封装测试。

分两类：
- 不需要网络/模型下载的：惰性加载、参数校验——这些是主测试套件的一部分。
- 需要真的下载模型跑一次转写的：默认跳过（CI 不该依赖网络下载几百 MB 模型
  这种不稳定、慢的前置条件），设环境变量 ``RTSE_RUN_ASR_INTEGRATION=1``
  才会跑，供本地手动验证真实转写效果。
"""

import os

import numpy as np
import pytest

from rtse import SAMPLE_RATE
from rtse.asr.whisper_engine import WhisperASR

RUN_INTEGRATION = os.environ.get("RTSE_RUN_ASR_INTEGRATION") == "1"


def test_model_is_not_loaded_until_first_use():
    """惰性加载：构造实例不该触发下载/加载，用不到 ASR 的代码路径不该为它付这个代价。"""
    engine = WhisperASR()
    assert engine._model is None


def test_rejects_wrong_sample_rate_without_loading_model():
    """采样率校验必须在真正碰模型之前就拦下来——错误的采样率会被静默地
    当成 16kHz 处理，转写结果全错但不会报错，比直接拒绝更难排查。
    """
    engine = WhisperASR()
    with pytest.raises(ValueError, match="16000"):
        engine.transcribe(np.zeros(8000), sr=8000)
    assert engine._model is None, "校验应该在加载模型之前就失败"


@pytest.mark.skipif(not RUN_INTEGRATION, reason="需要下载模型+联网，设 RTSE_RUN_ASR_INTEGRATION=1 手动跑")
@pytest.mark.parametrize("model_size,max_cer", [("tiny", 0.7), ("small", 0.3)])
def test_real_transcription_on_local_testset_sample(model_size, max_cer):
    """真实下载模型，对本地测试集里几条干净语音转写，确认返回的文本跟参考转写
    有实质重合。两档模型分开测、各自设合理阈值——用项目实际默认档位（small）
    验证真实评测会得到的质量水平；用最小档位（tiny）确认链路本身没坏，
    不能指望它出好结果（Whisper 的中文能力对模型规模非常敏感，
    tiny 档在中文上明显弱于同规模的英文表现）。

    用**多条样本取平均**而不是单条——单条样本可能刚好碰到生僻词/专有名词
    （比如"硬骨头六连"这种部队番号，任何规模的通用模型都容易在这类词上失手），
    平均几条更能反映真实水平，不会被一个难例带偏。
    """
    import json
    from pathlib import Path

    from rtse.asr.scoring import cer
    from rtse.audio.io import read_audio

    root = Path("data/testsets/aishell_controlled")
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    # 使用V1的clean恒等格 + 15 dB低混响格；不再依赖已删除的旧单测试集字段。
    recs = [
        r for r in idx["records"]
        if r.get("text") and (
            r["snr"] == "clean"
            or (isinstance(r["snr"], int) and r["snr"] >= 15 and r["rt60_bucket"] <= 0.2)
        )
    ][:5]
    assert recs, "测试集里没找到符合条件的样本"

    engine = WhisperASR(model_size=model_size)
    cers = []
    for rec in recs:
        clean = read_audio(root / rec["clean"])
        result = engine.transcribe(clean, sr=SAMPLE_RATE)
        assert result.text, "干净语音在高 SNR 下转写不该是空文本"
        assert result.language == "zh"
        c = cer(rec["text"], result.text)
        cers.append(c)
        print(f"\n[{model_size}] 参考: {rec['text']}\n      识别: {result.text}\n      CER={c:.3f}")

    mean_cer = sum(cers) / len(cers)
    print(f"\n[{model_size}] {len(cers)} 条样本平均 CER = {mean_cer:.3f}")
    assert mean_cer < max_cer, f"{model_size} 模型平均 CER {mean_cer:.3f} 超出预期 {max_cer}"
