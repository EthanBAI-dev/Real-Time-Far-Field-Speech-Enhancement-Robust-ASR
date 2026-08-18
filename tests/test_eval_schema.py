"""评测 CLI 与测试集 schema 的一致性测试。

**由来**：换数据集时生成端（notebook）的字段改了（`noise`→`noise_kind`、
`t60`→`rt60_nominal`/`rt60_measured`），但消费端（`rtse-eval`）没跟上，
直到跑全量评测时才以 `KeyError: 'noise'` 暴露——那时已经白等了模型训练和数据回传。
见 docs/ISSUES.md I-29。

这类"两端各自演进、中间靠字段名隐式耦合"的问题，靠人记是记不住的，
只能让测试在字段对不上时立刻失败。
"""

import json
from pathlib import Path

import pytest

from rtse.cli.evaluate import STRATA, _cell, _stratified

TESTSET = Path("data/testset/index.json")


def test_cell_rejects_records_missing_strata():
    """字段缺失必须**报错**，不能给默认值。

    给默认值的话，schema 不匹配会悄悄退化成"所有样本挤在同一格"，
    分层抽样失去意义、抽出来的 CER 只反映某一种条件，而且完全看不出异常。
    """
    with pytest.raises(KeyError, match="缺少字段"):
        _cell({"id": "x", "snr": 0})


def test_stratified_spreads_across_cells():
    """分层抽样必须覆盖到每个格子，而不是集中在前几格。"""
    recs = [{"id": f"{i}", "snr": s, "noise_kind": n, "rir_kind": r}
            for i, (s, n, r) in enumerate(
                [(s, n, r) for s in (-5, 0, 5) for n in ("stationary", "nonstationary")
                 for r in ("synth", "real")] * 3)]
    got = _stratified(recs, 2)
    assert len(got) == 12 * 2
    assert len({_cell(r) for r in got}) == 12


@pytest.mark.skipif(not TESTSET.exists(), reason="需要 data/testset（Colab 产物）")
def test_real_testset_has_every_field_the_evaluator_reads():
    """**真实测试集**必须带齐评测要读的字段。

    这是本文件的核心：不测构造出来的假数据，测磁盘上那份真的，
    因为出问题的正是"真实数据的 schema 变了"。
    """
    records = json.loads(TESTSET.read_text(encoding="utf-8"))["records"]
    assert records
    for k in STRATA + ("id", "clean", "noisy", "text"):
        missing = [r["id"] for r in records if k not in r]
        assert not missing, f"{len(missing)} 条记录缺少字段 {k!r}，例如 {missing[:3]}"


@pytest.mark.skipif(not TESTSET.exists(), reason="需要 data/testset（Colab 产物）")
def test_real_testset_records_both_nominal_and_measured():
    """标称值与实测值都要在。

    踩过两次：I-22（标称 RT60 不等于实测）和 I-24（标称 SNR 不等于实测）。
    结论都是**标称是请求、实测才是事实**，报告要引用后者，
    所以两个字段都必须存在，缺一不可。
    """
    records = json.loads(TESTSET.read_text(encoding="utf-8"))["records"]
    for pair in (("rt60_nominal", "rt60_measured"), ("snr", "snr_measured")):
        for k in pair:
            assert all(k in r for r in records), f"缺少 {k!r}——见 ISSUES.md I-22 / I-24"
