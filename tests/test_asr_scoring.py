"""CER/WER 计算测试。"""

import pytest

from rtse.asr.scoring import cer, wer


def test_cer_identical_text_is_zero():
    assert cer("我爱北京天安门", "我爱北京天安门") == 0.0


def test_cer_counts_character_edits():
    """7 个字错 1 个 -> 1/7，不是整句判定对/错。"""
    assert cer("我爱北京天安门", "我爱北京天按门") == pytest.approx(1 / 7)


def test_cer_normalizes_before_comparing():
    """参考用中文数字+无标点，识别结果用阿拉伯数字+标点——归一化后应该完全一致。"""
    assert cer("七十年代末我外出求学", "70年代末，我外出求学。") == 0.0


def test_cer_empty_reference():
    assert cer("", "") == 0.0
    assert cer("", "有内容") == 1.0


def test_cer_completely_wrong():
    assert cer("我爱北京天安门", "今天天气真好啊") > 0.5


def test_wer_meaningless_on_chinese_without_normalization_gap():
    """把这个已知陷阱钉成测试：中文没有空格，jiwer.wer 会把整句当一个"词"，
    结果只能是 0 或 1，跟真实识别质量无关——中文场景不该用这个函数的结果
    做任何定量判断，只能用 cer()。这条测试存在的意义是防止将来有人
    看错文档、拿 wer() 去测中文数据。
    """
    # 只错一个字，CER 应该很小，但 WER 必然是满分 1.0——这就是"没有信息量"的证据
    assert cer("我爱北京天安门", "我爱北京天按门") < 0.2
    assert wer("我爱北京天安门", "我爱北京天按门") == 1.0


def test_wer_is_meaningful_on_space_delimited_english():
    """英文（跨语言对照用）是空格分词的，WER 在这里是有意义的。"""
    assert wer("the quick brown fox", "the quick brown fox") == 0.0
    assert wer("the quick brown fox", "the quick brown dog") == pytest.approx(1 / 4)
