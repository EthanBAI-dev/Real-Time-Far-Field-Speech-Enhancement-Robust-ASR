"""把项目代码打包成一个 zip，供上传到 Google Drive 后在 Colab 里安装。

**为什么用打包而不是 git clone**：这个仓库目前不在 GitHub 上。
打包上传是最直接的路径，而且能精确控制传什么 —— 只传代码（约几百 KB），
不传 data/、models/、.venv/（那些几百 MB 到几 GB，传上去毫无意义）。

用法::

    uv run python scripts/pack_for_colab.py

产物：``dist/rtse-colab.zip``，上传到你在 Colab notebook 里配置的
``DRIVE_ROOT`` 目录下即可（见 ``notebooks/`` 配置 cell 与 ``docs/COLAB_GUIDE.md``）。
代码改动后重新跑一次、重新上传，Colab 里重新安装。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "rtse-colab.zip"

# 只打这些。刻意用白名单而不是黑名单 ——
# 黑名单总会漏掉新出现的大目录，白名单不会。
INCLUDE_FILES = ["pyproject.toml", "README.md"]
INCLUDE_DIRS = ["src", "configs", "tests"]
INCLUDE_DOCS = ["PLAN.md", "COLAB_GUIDE.md", "FINDINGS.md", "ISSUES.md"]

SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
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

    size_kb = OUT.stat().st_size / 1024
    print(f"已打包 {n} 个文件 → {OUT}  ({size_kb:.0f} KB)")
    print("\n下一步：把它上传到你 Colab notebook 里配置的 DRIVE_ROOT 目录下，")
    print("然后在 Colab 里按 docs/COLAB_GUIDE.md 执行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
