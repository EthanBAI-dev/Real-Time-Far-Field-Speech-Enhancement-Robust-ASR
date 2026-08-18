"""静态校验仓库中的 Colab notebooks。

只移除真正位于行首的 IPython ``!``/``%`` magic 及其反斜杠续行；不能因为
普通 Python 字符串或注释里出现 ``!`` 就跳过整个 cell。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _plain_python(source: str) -> str:
    out: list[str] = []
    in_magic_continuation = False
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if in_magic_continuation:
            in_magic_continuation = line.rstrip().endswith("\\")
            continue
        if stripped.startswith(("!", "%")):
            in_magic_continuation = line.rstrip().endswith("\\")
            out.append("\n")
            continue
        out.append(line)
    return "".join(out)


def main() -> int:
    failures = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        code_cells = 0
        for index, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            code_cells += 1
            source = "".join(cell.get("source", []))
            try:
                compile(_plain_python(source), f"{path.name}:cell-{index}", "exec")
            except SyntaxError as exc:
                failures.append(f"{path.name} cell {index}: {exc}")
        print(f"OK {path.name}: {len(nb.get('cells', []))} cells / {code_cells} code")
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
