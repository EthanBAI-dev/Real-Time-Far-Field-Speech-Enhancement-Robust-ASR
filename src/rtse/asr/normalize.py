"""CER 计算前的文本归一化。

**为什么这个模块必须做到位**：CER 是字符编辑距离，任何格式上的不一致都会被算成
"识别错误"，跟真实识别质量无关。本地测试集（THCHS-30）的参考转写是**无标点、
无空格、数字全部写成中文数字**的连续汉字（比如"七十年代""二月四日"），
而 Whisper 的输出习惯性带标点、可能用全角字符、数字倾向输出成阿拉伯数字——
不把这些差异抹平，CER 数字里绝大部分"错误"根本不是识别错误，是格式不匹配，
测出来的指标就是噪声，不能用来说明降噪有没有用。

流程（每一步都在解决一个具体的、真实存在的格式不一致）：
1. **繁体 → 简体**——第一次跑真实模型时才发现的坑：Whisper 训练数据里混了大量
   繁体中文（港台来源），转写普通话语音时偶尔会吐出繁体字（"戰鬥英雄"），
   而 THCHS-30 的参考转写是简体（"战斗英雄"）——不转换的话，这类"识别其实是对的，
   只是用了繁体字"的情况会被 CER 当成识别错误，而且是系统性的，不是偶发噪声
2. Unicode NFKC 正规化——把全角字母数字标点（Ａ１，）折叠成半角
3. **中文数字 → 阿拉伯数字**——方向选这边是因为参考文本固定用中文数字，
   反过来（阿拉伯转中文）在"二零二六"（逐字读）和"两千零二十六"（按数值读）
   之间是有歧义的，没有上下文猜不出该用哪种，而中文数字转数值是确定的、
   无歧义的运算
4. 剥离标点（中英文标点都要）
5. 剥离全部空白（CER）或折叠成单个空格（WER）——见下方两个函数的区别
6. 英文字母小写

繁简转换用 `opencc-python-reimplemented`（纯 Python 实现，不用像 `opencc` 原版
那样在 Windows 上编译 C++ 扩展）而不是自己维护映射表——这是字符集转换表，
不是本项目要展示的算法能力，跟"自己写 MCRA/CRN"不是一回事，属于合理的工具复用。
"""

from __future__ import annotations

import re
import string
import unicodedata

from opencc import OpenCC

__all__ = ["normalize_cer_text", "normalize_wer_text", "chinese_numeral_to_arabic"]

_T2S = OpenCC("t2s")

# ---------------------------------------------------------------------- 中文数字转换

_DIGIT = {
    "零": 0, "〇": 0, "一": 1, "壹": 1, "二": 2, "贰": 2, "两": 2,
    "三": 3, "叁": 3, "四": 4, "肆": 4, "五": 5, "伍": 5,
    "六": 6, "陆": 6, "七": 7, "柒": 7, "八": 8, "捌": 8, "九": 9, "玖": 9,
}
_UNIT = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_SECTION = {"万": 10_000, "亿": 100_000_000}
_NUMERAL_CHARS = set(_DIGIT) | set(_UNIT) | set(_SECTION)

# 一个数字游程里如果出现这些字符，判定为"按数值读"（磁场式），否则判定为
# "逐字读"（年份/编号式，比如"二零二六"）。只有零一二三四五六七八九本身
# 不足以区分——"二零二六"和"二"都只含数字字符，靠有没有十/百/千/万/亿判断。
_MAGNITUDE_MARKERS = set(_UNIT) | set(_SECTION)


def _parse_section(s: str) -> int:
    """解析一段不含 万/亿 的数字串（可能含 十/百/千），返回数值。

    标准算法：十/百/千 是"用前一个数字乘以这个量级"；十前面没有数字时
    （比如"十二"里的十），隐含乘数是 1，即 10*1+2=12，不是被错误解析成
    "空白*10"那样的 0。
    """
    total = 0
    current = 0
    for ch in s:
        if ch in _DIGIT:
            current = _DIGIT[ch]
        elif ch in _UNIT:
            total += (current if current else 1) * _UNIT[ch]
            current = 0
    return total + current


def _parse_magnitude_number(s: str) -> int:
    """解析一段含 万/亿 量级标记的数字串，返回数值。

    按"亿"→"万"→个位 三段切分，每段内部用 `_parse_section` 处理，
    段与段之间按量级相加。这是标准的中文数字解析算法，能正确处理
    "三千零五"（3005，零起占位作用）、"两万零一"（20001）这类带零的写法——
    因为零不参与 `_parse_section` 的乘法运算，只在游程扫描时被跳过，
    天然不会引入错误的量级。
    """
    total = 0
    rest = s
    for marker, scale in (("亿", 100_000_000), ("万", 10_000)):
        if marker in rest:
            left, _, rest = rest.partition(marker)
            total += (_parse_section(left) if left else 1) * scale
    total += _parse_section(rest)
    return total


def _convert_run(run: str) -> str:
    """转换一段连续的数字字符游程。按有没有量级标记分两种情况处理。"""
    if any(ch in _MAGNITUDE_MARKERS for ch in run):
        return str(_parse_magnitude_number(run))
    # 纯 零~九 序列（可能含"两"）：逐字读，比如年份"二零二六" -> "2026"。
    # "两"在逐字读语境下不常见，但仍按 2 处理，不引入特殊分支。
    return "".join(str(_DIGIT[ch]) for ch in run)


def chinese_numeral_to_arabic(text: str) -> str:
    """把文本里的中文数字游程转换成阿拉伯数字。非数字字符原样保留。

    只处理**连续的**数字字符游程——"三个五"（三 个 五，中间被"个"打断）
    会被拆成两段分别转换（"3个5"），不会被误判成跨词的单个数字，
    这是正确行为：日常语言里数字通常不会跨着量词连读。
    """
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] in _NUMERAL_CHARS:
            j = i
            while j < n and text[j] in _NUMERAL_CHARS:
                j += 1
            out.append(_convert_run(text[i:j]))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------- 整体归一化

# 标点：用 unicodedata 的 Punctuation 大类兜底，再补几个类别没盖到但
# 中文文本常见的符号（书名号、间隔号等）。
_EXTRA_PUNCT = "、。，·「」『』〈〉《》【】〔〕（）—…‘’“”"


def _strip_punct(text: str) -> str:
    return "".join(
        ch for ch in text
        if not (unicodedata.category(ch).startswith("P") or ch in _EXTRA_PUNCT or ch in string.punctuation)
    )


def _normalize_common(text: str) -> str:
    """CER 和 WER 共用的前三步：全角折叠、中文数字转换、去标点、转小写。

    **空白处理故意不放在这里**——中文 CER 按字符算，空白毫无意义、应该全部
    去掉；英文 WER 按空格分词，空白正是词边界、绝不能去掉。这两种需求互斥，
    一开始把空白处理也塞进这个共用函数里，导致 `wer()` 把英文单词也拼成了
    一整个字符串，WER 恒等于 0 或 1（跟 `test_wer_meaningless_on_chinese_...`
    描述的中文那个坑一模一样，只是这次是我自己引入到英文路径上的）。
    """
    text = _T2S.convert(text)  # 繁体 -> 简体
    text = unicodedata.normalize("NFKC", text)  # 全角 -> 半角
    text = chinese_numeral_to_arabic(text)
    text = _strip_punct(text)
    return text.lower()


def normalize_cer_text(text: str) -> str:
    """CER 归一化：中文按字符算，字与字之间的空白没有意义，全部去掉。"""
    return re.sub(r"\s+", "", _normalize_common(text))


def normalize_wer_text(text: str) -> str:
    """WER 归一化：空白是单词边界，只把连续空白折叠成单个空格、掐头去尾，
    绝不能整个删掉——删掉的话所有单词会拼成一个大"词"，WER 只会是 0 或 1。
    """
    return re.sub(r"\s+", " ", _normalize_common(text)).strip()
