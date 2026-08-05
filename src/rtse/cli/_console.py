"""控制台输出的编码防护。

**本机的控制台代码页是 cp932（日文 Shift-JIS）**，不是 UTF-8 也不是 GBK。
在这台机器上，一个普通的 `print("实时演示")` 会直接抛
``UnicodeEncodeError: 'cp932' codec can't encode character``，
整个命令连启动都启动不了（见 docs/ISSUES.md I-11）。

两层防护：

1. ``sys.stdout/stderr`` 重配为 UTF-8 且 ``errors="replace"`` ——
   保证**永远不会因为编码而崩溃**，最坏情况只是显示成问号。
2. 所有面向用户的输出统一走 ``rich.Console``。rich 在 Windows 上走的是
   控制台 API 而不是字节流，能正确渲染非 ASCII 字符，不受代码页影响。

第 1 层是保命的（防崩溃），第 2 层是保效果的（防乱码）。两层都要有：
第三方库里的 print 我们管不到，只有第 1 层能兜住。
"""

from __future__ import annotations

import sys

from rich.console import Console

__all__ = ["console", "harden_stdio"]


def harden_stdio() -> None:
    """把标准输出/错误重配为 UTF-8，避免非 UTF-8 代码页下的编码崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # 被重定向到非 TextIOWrapper 的对象（如 pytest 捕获流）
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 流已关闭或不支持重配。不是致命问题，继续跑。
            pass


harden_stdio()

#: 所有 CLI 共用的输出通道。
console = Console()
