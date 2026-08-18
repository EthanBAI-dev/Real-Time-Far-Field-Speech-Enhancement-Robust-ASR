"""把「每次要传到 Colab 的东西」打包成一个统一文件夹 + 一个 zip。

**为什么要统一**：之前代码包在 `dist/rtse-colab.zip`，notebook 在
`notebooks/*.ipynb`，是两个不同目录。每次代码或 notebook 改了，
都要分别去两个地方找文件、分两次上传，还容易漏传（上一次踩过的坑：
notebook 传了新的，代码包忘了重新打包，Colab 跑的是新 notebook + 旧代码）。

现在这个脚本把两者一起收进 ``dist/colab_upload/``：

- ``rtse-colab.zip``：代码包（白名单打包 src/configs/tests + 关键文档）；
- ``rtse_colab.ipynb``：**三册合并后的单一 notebook**，由
  ``scripts/build_merged_notebook.py`` 生成。``notebooks/`` 下的三个分册
  仍是编辑源，但不再上传——Colab 每个会话都会清空 ``/content``，
  开三个 notebook 就要付三次「解压代码 + 解压 20GB 语料」，一次约 20 分钟。

再把整个 ``dist/colab_upload/`` 压成一个 ``dist/colab_upload.zip``，
方便一次性传输（比如通过聊天发送）。

**Drive 上的目录约定**：解开 zip 后，把里面两个文件传到
``<DRIVE_ROOT>/colab_upload/`` 下。Colab 产出的东西全部落在
``<DRIVE_ROOT>/colab_outputs/``（要下载结果就整个下这一个目录）。
``archives/`` 那约 20GB 的语料缓存**位置不动**——搬走会让 ``fetch_dns``
的下载标记全部失效，得重下一遍。

用法::

    uv run python scripts/pack_for_colab.py

代码或 notebook 改动后重新跑一次、重新上传，Colab 里重新安装。
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "dist" / "colab_upload"
CODE_ZIP = BUNDLE_DIR / "rtse-colab.zip"
BUNDLE_ZIP = ROOT / "dist" / "colab_upload.zip"
NOTEBOOKS_SRC = ROOT / "notebooks"
# 只发合并后的这一个。三个分册留在 notebooks/ 作为编辑源，不再上传——
# Colab 的 /content 每会话清空，开三个 notebook 就要付三次「解压代码 + 解压 20GB 语料」，
# 一次约 20 分钟。合并版由 scripts/build_merged_notebook.py 生成。
NOTEBOOK_NAMES = ("rtse_colab.ipynb",)

# 只打这些。刻意用白名单而不是黑名单 ——
# 黑名单总会漏掉新出现的大目录，白名单不会。
INCLUDE_FILES = ["pyproject.toml", "README.md"]
INCLUDE_DIRS = ["src", "configs", "tests"]
INCLUDE_DOCS = ["PLAN.md", "COLAB_GUIDE.md", "DATASETS_V1.md", "FINDINGS.md", "ISSUES.md"]

SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"}


def _pack_code() -> int:
    n = 0
    with zipfile.ZipFile(CODE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in INCLUDE_FILES:
            p = ROOT / rel
            if p.exists():
                z.write(p, rel)
                n += 1
        for d in INCLUDE_DIRS:
            for p in (ROOT / d).rglob("*"):
                if p.is_file() and not (SKIP_PARTS & set(p.parts)):
                    z.write(p, str(p.relative_to(ROOT)))
                    n += 1
        for name in INCLUDE_DOCS:
            p = ROOT / "docs" / name
            if p.exists():
                z.write(p, f"docs/{name}")
                n += 1
    return n


def _copy_notebooks() -> list[str]:
    copied = []
    missing = []
    for name in NOTEBOOK_NAMES:
        src = NOTEBOOKS_SRC / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, BUNDLE_DIR / name)
        copied.append(name)
    if missing:
        # notebook 没传全，去 Colab 打开的就是缺了这一份 —— 必须在这里就炸出来，
        # 不能让脚本"看起来打包成功"但其实少了一个文件。
        raise FileNotFoundError(f"notebooks/ 下缺少 {missing}，检查文件名是否变了")
    return copied


def _zip_bundle() -> None:
    if BUNDLE_ZIP.exists():
        BUNDLE_ZIP.unlink()
    with zipfile.ZipFile(BUNDLE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(BUNDLE_DIR.iterdir()):
            if p.is_file():
                z.write(p, p.name)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)

    n_code = _pack_code()
    code_kb = CODE_ZIP.stat().st_size / 1024
    print(f"代码包：{n_code} 个文件 → {CODE_ZIP}  ({code_kb:.0f} KB)")

    copied = _copy_notebooks()
    print(f"Notebook：{len(copied)} 个 → {BUNDLE_DIR}/  ({', '.join(copied)})")

    _zip_bundle()
    bundle_kb = BUNDLE_ZIP.stat().st_size / 1024
    print(f"\n统一打包 → {BUNDLE_ZIP}  ({bundle_kb:.0f} KB)")
    print(f"（也可以直接用 {BUNDLE_DIR}/ 这个文件夹，两者内容一样）")
    print("\n下一步：解开这个 zip，把里面两个文件传到 Drive 的")
    print("  <DRIVE_ROOT>/colab_upload/   （没有就新建这个目录）")
    print("Colab 的产出会全部落在 <DRIVE_ROOT>/colab_outputs/，下载就下那一个目录。")
    print("archives/ 那约 20GB 语料缓存位置不动，别搬。详见 docs/COLAB_GUIDE.md。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
