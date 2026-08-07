"""CER 归一化测试。

这是整个 ASR 集成里最容易悄悄出错、又最容易被忽视的一环——归一化做得不对，
CER 数字看着挺好看，其实测的是格式差异不是识别质量。`docs/PLAN.md` 里专门把
"中文 CER 归一化不彻底"列成一条风险，这里用手工构造的边界样例逐条堵上。
"""

import pytest

from rtse.asr.normalize import chinese_numeral_to_arabic, normalize_cer_text

# ----------------------------------------------------------------- 中文数字：逐字读


@pytest.mark.parametrize(
    "text,expected",
    [
        ("二零二六", "2026"),
        ("一九九零", "1990"),
        ("零", "0"),
        ("二零二六年四月", "2026年4月"),  # 后面的"四月"是量级读法，跟前面的年份分开处理
    ],
)
def test_digit_by_digit_reading(text, expected):
    """年份/编号这类场景，每个字对应一个数位，不做进位运算。"""
    assert chinese_numeral_to_arabic(text) == expected


# ----------------------------------------------------------------- 中文数字：按数值读


@pytest.mark.parametrize(
    "text,expected",
    [
        ("十", "10"),
        ("十一", "11"),
        ("十二", "12"),  # 关键边界：不能被误解析成"1"+"10"+"2"=112
        ("二十", "20"),
        ("二十一", "21"),
        ("九十九", "99"),
        ("一百", "100"),
        ("一百零一", "101"),  # 零起占位作用，不能被吃掉或误加值
        ("一百一十", "110"),
        ("两百", "200"),  # "两"在量级语境下等价于"二"
        ("三千零五", "3005"),
        ("一千二百三十四", "1234"),
        ("一万", "10000"),
        ("两万零一", "20001"),
        ("一亿", "100000000"),
        ("一亿两千万", "120000000"),
    ],
)
def test_magnitude_reading(text, expected):
    """量级读法（含十/百/千/万/亿）必须按标准算法解析出正确数值。"""
    assert chinese_numeral_to_arabic(text) == expected


def test_numeral_run_is_interrupted_by_non_numeral_char():
    """"三个五"中间被"个"打断，应该拆成两段分别转换，不能跨字符误连成一个数。"""
    assert chinese_numeral_to_arabic("三个五") == "3个5"


def test_numeral_conversion_preserves_surrounding_text():
    assert chinese_numeral_to_arabic("七十年代末我外出求学") == "70年代末我外出求学"
    assert chinese_numeral_to_arabic("二月四日住进新西门") == "2月4日住进新西门"


def test_lone_liang_without_magnitude_marker():
    """"两"单独出现（没有跟着量级字）时也应该转成 2，不是保持原样或报错。"""
    assert chinese_numeral_to_arabic("两个人") == "2个人"


# ----------------------------------------------------------------- 完整归一化管线


def test_traditional_to_simplified():
    """真实模型跑出来的坑：Whisper 训练数据里混了繁体中文，转写普通话时
    偶尔吐出繁体字，参考转写是简体——这类"识别是对的，只是用了繁体字"
    不该被算成识别错误。
    """
    assert normalize_cer_text("戰鬥英雄和轉業戰士") == normalize_cer_text("战斗英雄和转业战士")


def test_fullwidth_to_halfwidth():
    assert normalize_cer_text("ＡＢＣ１２３") == "abc123"


def test_strips_chinese_and_english_punctuation():
    assert normalize_cer_text("你好，世界！") == "你好世界"
    assert normalize_cer_text("Hello, world!") == "helloworld"
    assert normalize_cer_text("「引用」·间隔号") == "引用间隔号"


def test_strips_all_whitespace():
    """中文 CER 按字符算，字与字之间的空格没有意义，必须全部去掉。"""
    assert normalize_cer_text("我 爱 北 京") == "我爱北京"
    assert normalize_cer_text("我\t爱\n北京") == "我爱北京"


def test_lowercases_english():
    assert normalize_cer_text("Hello World") == "helloworld"


def test_empty_and_punctuation_only_input():
    assert normalize_cer_text("") == ""
    assert normalize_cer_text("，。！？") == ""


def test_real_thchs30_style_sentence_is_unchanged_when_no_numbers_present():
    """真实测试集里的转写本来就无标点无空格，归一化不应该破坏这类已经干净的文本。

    挑一句**不含任何数字字符**的真实样本——"四月"这种带数字的句子归一化后
    "四"会被转成"4"，那是归一化**正确**在做它该做的事，不该拿来测"保持不变"。
    """
    text = "君子多欲则贪慕富贵枉道速祸小人多欲则多求妄用败家丧身"
    assert normalize_cer_text(text) == text


def test_numeral_conversion_is_intentional_not_a_bug():
    """反过来验证：含数字的真实样本，归一化**应该**把中文数字转成阿拉伯数字——
    这正是这个模块存在的目的，不是"归一化不该动数据"的例外。
    """
    text = "四月的林峦更是绿得鲜活秀媚诗意盎然"
    assert normalize_cer_text(text) == "4月的林峦更是绿得鲜活秀媚诗意盎然"


def test_reference_and_whisper_style_output_converge():
    """核心场景：参考文本用中文数字，Whisper 输出用阿拉伯数字+标点，
    归一化后两者应该完全一致——这是整个归一化管线存在的意义。
    """
    reference = "七十年代末我外出求学母亲叮咛我"
    whisper_like_output = "70年代末，我外出求学，母亲叮咛我。"
    assert normalize_cer_text(reference) == normalize_cer_text(whisper_like_output)
