"""项目路径常量。

集中定义的原因：C: 盘只剩 18 GB（见 docs/ISSUES.md I-04），
数据与模型目录随时可能需要整体迁到别的盘。所有路径只在这里写一次，
迁移时改环境变量即可，不用全仓库搜替换。

可用环境变量覆盖：``RTSE_DATA_DIR`` / ``RTSE_MODELS_DIR`` / ``RTSE_RESULTS_DIR``
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PROJECT_ROOT", "DATA_DIR", "MODELS_DIR", "RESULTS_DIR", "CONFIG_DIR", "ensure_dirs"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dir(env: str, default: str) -> Path:
    return Path(os.environ.get(env, PROJECT_ROOT / default)).resolve()


DATA_DIR = _dir("RTSE_DATA_DIR", "data")
MODELS_DIR = _dir("RTSE_MODELS_DIR", "models")
RESULTS_DIR = _dir("RTSE_RESULTS_DIR", "results")
CONFIG_DIR = PROJECT_ROOT / "configs"

# 本地磁盘硬预算（GB）。doctor 会检查剩余空间并在低于阈值时告警。
LOCAL_DISK_BUDGET_GB = 5.0
LOCAL_DISK_WARN_FREE_GB = 10.0


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
