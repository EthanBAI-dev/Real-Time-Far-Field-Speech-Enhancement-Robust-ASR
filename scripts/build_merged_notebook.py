"""把三个 Colab notebook 合并成一个 `rtse_colab.ipynb`。

**为什么合并**：三个 notebook 各自带一份完全相同的 4 个初始化 cell
（挂载 Drive → 配置 → 解压安装代码 → 信号链路自检）。Colab 的 `/content`
每个会话都会清空，所以每开一个 notebook 就要重跑一遍安装 + 重新解压 20GB 语料，
一次约 20 分钟，跑完整条链路要付三次。合并成一个之后只付一次。

顺带去掉的重复：`02_train` 里有一份 `fetch_dns` 的 **196 行完整复制**
（跟 `01_data_prep` 逐字相同）。它当初存在只是因为 02 要能独立运行；
合并后 PART 1 已经在同一个会话里定义过并下载好了，这份副本可以整个删掉。
这也正是 I-25 那个 bug 当初需要在两个文件里各修一遍的原因。

**Drive 目录约定**（本脚本会改写配置 cell）：

    MyDrive/Audio AI/RTSE/
    ├── colab_upload/      ← 你上传的代码包与 notebook（本地 dist/colab_upload/ 的内容）
    ├── colab_outputs/     ← 所有产物，要下载就整个下这一个目录
    │   ├── checkpoints/  models/  testsets/  logs/
    │   └── manifest.json  training_gates.json  colab_metrics.json  rtse_handoff.zip
    └── archives/          ← 压缩包缓存，**位置不动**

`archives/` 刻意保持在原位：那里是已经下好的约 20GB 语料，换目录会让
`fetch_dns` 的下载标记全部失效、白下一遍。

用法::

    uv run python scripts/build_merged_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
OUT_NB = NB_DIR / "rtse_colab.ipynb"

# 每个 notebook 的前 N 个 cell 是初始化（挂载/配置/安装/自检），合并后只保留一份。
SETUP_CELLS = {"01_data_prep": 5, "02_train": 5, "03_export_eval": 4}
# 合并后可以整个删掉的重复 cell（下标基于原 notebook）。
DROP_CELLS = {"02_train": [7]}  # fetch_dns 的完整副本，PART 1 已定义

PART_HEADERS = [
    (
        "01_data_prep",
        """---

# ═══════════════════════════════════════════════════════════════════
# PART 1 · 数据准备
# ═══════════════════════════════════════════════════════════════════

下载 DNS5 / AISHELL-1 / WenetSpeech，建清单、按平稳性给噪声分组，
生成三套职责分离的 V1 测试集。

**耗时最长的一段**：首次要下约 20 GB（之后走 `archives/` 缓存），
每个新会话都要重新解压到 `/content`（Drive 是 FUSE，训练直接读会很慢）。""",
    ),
    (
        "02_train",
        """---

# ═══════════════════════════════════════════════════════════════════
# PART 2 · 模型训练
# ═══════════════════════════════════════════════════════════════════

在线动态混音训练 CRN，跑完后过**无害性与有效性闸门**。

⚠️ 闸门不过就不要往下走 PART 3：导出一个连干净输入都会破坏的模型没有意义。""",
    ),
    (
        "03_export_eval",
        """---

# ═══════════════════════════════════════════════════════════════════
# PART 3 · ONNX 导出 + 评测 + 打包回传
# ═══════════════════════════════════════════════════════════════════

流式 ONNX 导出与三层校验、DNS 客观质量评测（含 PESQ/DNSMOS），
最后打包成一个 `rtse_handoff.zip` 供下载。""",
    ),
]

# ── 配置 cell 的路径改写 ────────────────────────────────────────────────
CONFIG_REPLACEMENTS = [
    (
        """CKPT_DIR = f'{DRIVE}/checkpoints'   # 训练断点，每 epoch 保存
MODEL_DIR = f'{DRIVE}/models'       # 导出的 ONNX
TESTSETS_DIR = f'{DRIVE}/testsets'  # V1 三套职责分离的固定测试集""",
        """# ── 上传区与产物区分开 ─────────────────────────────────────────────────
# 你上传的东西都在 colab_upload/，Colab 产出的东西都在 colab_outputs/，
# 要下载结果就整个下 colab_outputs/ 这一个目录，不用在 Drive 根目录里挑。
UPLOAD_DIR = f'{DRIVE}/colab_upload'
OUT = f'{DRIVE}/colab_outputs'

CKPT_DIR = f'{OUT}/checkpoints'     # 训练断点，每 epoch 保存
MODEL_DIR = f'{OUT}/models'         # 导出的 ONNX
TESTSETS_DIR = f'{OUT}/testsets'    # V1 三套职责分离的固定测试集
MANIFEST = f'{OUT}/manifest.json'          # 数据清单
GATES = f'{OUT}/training_gates.json'       # 训练闸门结果
COLAB_METRICS = f'{OUT}/colab_metrics.json'  # Colab 侧指标（含 PESQ）""",
    ),
    ("LOG_DIR = f'{DRIVE}/logs'", "LOG_DIR = f'{OUT}/logs'"),
    (
        "for d in [WORK, ARCHIVE_DIR, DATA, CKPT_DIR, MODEL_DIR, TESTSETS_DIR, LOG_DIR]:",
        (
            "for d in [WORK, ARCHIVE_DIR, DATA, UPLOAD_DIR, OUT,\n"
            "          CKPT_DIR, MODEL_DIR, TESTSETS_DIR, LOG_DIR]:"
        ),
    ),
    (
        """print(f'  代码包(需手动上传)  {DRIVE}/rtse-colab.zip')
print(f'  压缩包缓存          {ARCHIVE_DIR}')
print(f'  语料解压目标        {DATA}')
print(f'  数据清单            {DRIVE}/manifest.json')
print(f'  三套固定测试集      {TESTSETS_DIR}')
print(f'  训练断点  ★         {CKPT_DIR}/<模型名>/{{last,best}}.pt')
print(f'  导出模型  ★         {MODEL_DIR}/<模型名>.onnx')""",
        """print(f'  代码包(需手动上传)  {UPLOAD_DIR}/rtse-colab.zip')
print(f'  压缩包缓存          {ARCHIVE_DIR}   ← 位置不动，别搬')
print(f'  语料解压目标        {DATA}')
print(f'  ── 以下全在 colab_outputs/，下载就下这一个目录 ──')
print(f'  数据清单            {MANIFEST}')
print(f'  三套固定测试集      {TESTSETS_DIR}')
print(f'  训练断点  ★         {CKPT_DIR}/<模型名>/{{last,best}}.pt')
print(f'  导出模型  ★         {MODEL_DIR}/<模型名>.onnx')""",
    ),
]

# ── 安装 cell 的路径改写 ────────────────────────────────────────────────
INSTALL_REPLACEMENTS = [
    (
        "ZIP = f'{DRIVE}/rtse-colab.zip'",
        "ZIP = f'{UPLOAD_DIR}/rtse-colab.zip'",
    ),
    (
        """# 需要手动上传到 DRIVE_ROOT 目录下。**代码改过就要重新上传**，""",
        """# 需要手动上传到 Drive 的 colab_upload/ 目录下。**代码改过就要重新上传**，""",
    ),
    (
        """    '请先在本地执行 `uv run python scripts/pack_for_colab.py`，'
    f'再把 dist/colab_upload/ 下的文件（zip + 3 个 notebook）传到 Drive 的 {DRIVE} 下。\\n'
    f'该目录下现有：{sorted(os.listdir(DRIVE))[:12]}'""",
        """    '请先在本地执行 `uv run python scripts/pack_for_colab.py`，'
    f'再把 dist/colab_upload/ 下的全部内容传到 Drive 的 {UPLOAD_DIR}/ 下。\\n'
    f'该目录下现有：{sorted(os.listdir(UPLOAD_DIR)) if os.path.isdir(UPLOAD_DIR) else "（目录不存在）"}'""",
    ),
]

# ── 各 PART 内部的产物路径改写 ──────────────────────────────────────────
BODY_REPLACEMENTS = [
    ("Path(f'{DRIVE}/manifest.json')", "Path(MANIFEST)"),
    ("print(f'\\n清单已写入 {DRIVE}/manifest.json')", "print(f'\\n清单已写入 {MANIFEST}')"),
    ("mf = Path(f'{DRIVE}/manifest.json')", "mf = Path(MANIFEST)"),
    ("Path(f'{DRIVE}/training_gates.json')", "Path(GATES)"),
    ("gate_path = Path(f'{DRIVE}/training_gates.json')", "gate_path = Path(GATES)"),
    ("Path(f'{DRIVE}/colab_metrics.json')", "Path(COLAB_METRICS)"),
    (
        "print(f'已写入 {len(rows)} 条指标 → {DRIVE}/colab_metrics.json')",
        "print(f'已写入 {len(rows)} 条指标 → {COLAB_METRICS}')",
    ),
    ("'colab_metrics': Path(DRIVE) / 'colab_metrics.json',", "'colab_metrics': Path(COLAB_METRICS),"),
    ("'data_manifest': Path(DRIVE) / 'manifest.json',", "'data_manifest': Path(MANIFEST),"),
    ("'training_gates': Path(DRIVE) / 'training_gates.json',", "'training_gates': Path(GATES),"),
    ("archive = Path(DRIVE) / 'rtse_handoff.zip'", "archive = Path(OUT) / 'rtse_handoff.zip'"),
    (
        '!cd "{DRIVE}" && rm -f testsets_v1.zip && zip -q -r testsets_v1.zip testsets && ls -lh testsets_v1.zip',
        '!cd "{OUT}" && rm -f testsets_v1.zip && zip -q -r testsets_v1.zip testsets && ls -lh testsets_v1.zip',
    ),
    (
        "print(f'  2. 下载 {DRIVE}/testsets_v1.zip，解压到本地 data/ 下')",
        "print(f'  2. 下载 {OUT}/testsets_v1.zip，解压到本地 data/ 下')",
    ),
    # 合并后 01/02/03 这些编号不存在了，指向它们的说明要跟着改，
    # 否则报错信息会让人去找一个根本没有的 notebook。
    (
        "# 正式导出必须先通过 02 的 clean透传 + 去噪正增益闸门。",
        "# 正式导出必须先通过 PART 2 的 clean透传 + 去噪正增益闸门。",
    ),
    (
        "'缺少 training_gates.json；先完整运行 02 的无害性与有效性闸门'",
        "'缺少 training_gates.json；先完整运行 PART 2 的无害性与有效性闸门'",
    ),
]

HEAD_MD = """# RTSE · Colab 全流程（数据准备 → 训练 → 导出评测）

三个原 notebook 合并成这一个。**合并的唯一目的是省时间**：
Colab 的 `/content` 每个会话都会清空，原来每开一个 notebook 就要重跑一遍
「解压代码包 + 解压 20GB 语料」，一次约 20 分钟，整条链路要付三次；
现在初始化只跑一次。

## 怎么用

1. 先跑 **初始化** 的 4 个 cell（挂载 → 配置 → 安装代码 → 自检）；
2. 然后按 PART 1 → 2 → 3 顺序往下跑。

三个 PART 之间用 `═══` 分割线隔开，每段开头都写了它负责什么。
**PART 2 的闸门不过就不要跑 PART 3**——导出一个连干净输入都会破坏的模型没有意义。

## Drive 目录约定

```
MyDrive/Audio AI/RTSE/
├── colab_upload/     ← 你上传的：rtse-colab.zip + 本 notebook
├── colab_outputs/    ← 所有产物，要下载就整个下这一个目录
│   ├── checkpoints/  models/  testsets/  logs/
│   └── manifest.json  training_gates.json  colab_metrics.json  rtse_handoff.zip
└── archives/         ← 已下好的约 20GB 语料缓存，**位置不动**
```

`archives/` 刻意留在原处：搬走会让下载标记全部失效，20GB 得重下一遍。

---

# ═══════════════════════════════════════════════════════════════════
# 初始化（只需跑一次）
# ═══════════════════════════════════════════════════════════════════
"""


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _apply(cell: dict, pairs: list[tuple[str, str]], where: str) -> int:
    """对 cell 做字符串替换，返回命中数。未命中的规则由调用方负责报错。"""
    src = "".join(cell["source"])
    hits = 0
    for old, new in pairs:
        if old in src:
            src = src.replace(old, new)
            hits += 1
    cell["source"] = src.splitlines(keepends=True)
    return hits


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    nbs = {name: json.loads((NB_DIR / f"{name}.ipynb").read_text(encoding="utf-8"))
           for name, _ in PART_HEADERS}

    merged: list[dict] = [_md(HEAD_MD)]

    # 初始化：取 01 的 cell 1..4（cell 0 是它自己的标题，丢弃）
    setup = [json.loads(json.dumps(c)) for c in nbs["01_data_prep"]["cells"][1:5]]
    n_cfg = _apply(setup[1], CONFIG_REPLACEMENTS, "config")
    n_inst = _apply(setup[2], INSTALL_REPLACEMENTS, "install")
    if n_cfg != len(CONFIG_REPLACEMENTS):
        raise SystemExit(f"配置 cell 改写只命中 {n_cfg}/{len(CONFIG_REPLACEMENTS)} 条，源文件可能变了")
    if n_inst != len(INSTALL_REPLACEMENTS):
        raise SystemExit(f"安装 cell 改写只命中 {n_inst}/{len(INSTALL_REPLACEMENTS)} 条，源文件可能变了")
    merged.extend(setup)

    # 三个 PART
    body_hits = 0
    for name, header in PART_HEADERS:
        merged.append(_md(header))
        skip = SETUP_CELLS[name]
        drop = set(DROP_CELLS.get(name, []))
        for i, cell in enumerate(nbs[name]["cells"]):
            if i < skip or i in drop:
                continue
            c = json.loads(json.dumps(cell))
            body_hits += _apply(c, BODY_REPLACEMENTS, name)
            merged.append(c)

    out = {
        "cells": merged,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT_NB.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    n_code = sum(1 for c in merged if c["cell_type"] == "code")
    print(f"已合并 → {OUT_NB}")
    print(f"  {len(merged)} 个 cell（code {n_code}，markdown {len(merged) - n_code}）")
    print(f"  产物路径改写命中 {body_hits} 处")
    print(f"  去掉重复的初始化 cell {sum(SETUP_CELLS.values()) - 4} 个"
          f"、fetch_dns 副本 {sum(len(v) for v in DROP_CELLS.values())} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
