# 开发指南

这是一个个人作品集项目，暂不特别设计外部协作流程，但开发环境和约定写在这里，
方便别人（或者未来的自己）看懂怎么跑起来、改哪里、按什么规矩改。

## 环境

用 [uv](https://docs.astral.sh/uv/) 管理，不用 conda、不用系统 pip。

```bash
uv sync --extra dev
uv run pytest
uv run rtse-doctor
```

Python 版本锁定 3.11（原因见 [`docs/ISSUES.md`](docs/ISSUES.md) I-02）。

## 测试

改代码前后都跑一遍：

```bash
uv run pytest
```

新增功能要配测试，尤其是这几类不写测试就等于没做完的改动：

- 任何涉及**因果性**的模型/信号处理改动 —— 必须有一条"改动未来输入不影响当前输出"的测试
- 任何"离线 vs 流式"应当一致的路径 —— 必须有数值一致性测试
- 任何 ONNX 导出的改动 —— 必须跑一遍三层校验（ONNX 流式 vs PyTorch 整段）

## 代码风格

用 ruff：

```bash
uv run ruff check src tests
```

CI 目前只让 ruff 报告问题、不拦截合并（`continue-on-error: true`），
因为代码还没被 ruff 通篇清过一遍。清完一轮后会去掉这个豁免。

## 遇到坑之后

这个项目有一份问题记录 [`docs/ISSUES.md`](docs/ISSUES.md)，每个真实踩到的坑都记录了
现象、根因、解法，而不只是"改了下就好了"。改代码时如果碰到类似的坑（尤其是
Windows 平台兼容性、STFT/dB 标定、Colab 环境差异这几类），照着这个格式续写一条，
好过什么都不写。

## 目录结构

见 [`README.md`](README.md) 末尾的项目结构说明，以及 [`docs/PLAN.md`](docs/PLAN.md)
的完整架构与阶段划分。
