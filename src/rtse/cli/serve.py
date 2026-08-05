"""``rtse-server`` —— 启动 Web 演示服务。"""

from __future__ import annotations

import argparse

import uvicorn

from rtse.cli._console import console
from rtse.paths import ensure_dirs


def main() -> int:
    p = argparse.ArgumentParser(description="启动 RTSE Web 演示服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="改代码自动重启（开发用）")
    args = p.parse_args()

    ensure_dirs()
    url = f"http://{args.host}:{args.port}"
    console.print(f"\n[bold]RTSE Web 演示[/bold]  →  [link={url}]{url}[/link]\n")
    console.print("  [dim]/[/dim]       实时麦克风演示")
    console.print("  [dim]/lab[/dim]    文件实验台（不需要麦克风）")
    console.print("  [dim]/bench[/dim]  基准看板")
    console.print(
        "\n[yellow]注意[/yellow]：浏览器只在 localhost 或 HTTPS 下才允许访问麦克风。\n"
        "用 [cyan]127.0.0.1[/cyan] 打开（不要用局域网 IP），否则麦克风按钮会无声失败。\n"
    )

    uvicorn.run("rtse.server.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
