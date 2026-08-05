# 环境勘察报告（本机）

> 勘察时间：2026-08-05
> 勘察方式：PowerShell 直接查询，未做任何安装或修改

---

## 1. 硬件

| 项目 | 实测值 |
|---|---|
| CPU | Intel Core i7-14700，20 核 / 28 线程 |
| 内存 | 32 GB（31.2 GiB 可见） |
| GPU | NVIDIA RTX 4000 Ada Generation |
| 显存 | 20475 MiB（约 20 GB），勘察时已占用 1573 MiB |
| 显卡驱动 | 572.16 |
| CUDA Runtime（驱动侧） | 12.8 |
| 操作系统 | Windows 11 Pro 10.0.26100 |

## 2. 磁盘

| 盘符 | 可用 | 性质 | 本项目可用性 |
|---|---|---|---|
| C: | **18.1 GB** | 系统盘，项目当前所在 | 可用但紧张，需要严格控制体积 |
| D: | 30.6 GB | 本地盘 | 备用，放大文件缓存 |
| S: | 155.6 GB | 本地盘 | 备用 |
| W: | 95.7 GB | 本地盘 | 备用 |
| V: / X: | 86.4 GB | 本地盘（同一卷） | 备用 |
| G: M: N: P: Q: Y: | 910 GB | **公司网络盘**（内含 3D-data / 図面 等业务目录） | **禁止使用**，不放任何项目数据 |

> **结论**：本地磁盘预算按 **≤ 5 GB** 设计。完整 DNS Challenge 数据集（数百 GB）绝无可能落地，
> 这正是"数据下载与训练放 Colab"方案成立的真正理由。

## 3. 软件工具链

| 工具 | 状态 | 路径 / 版本 |
|---|---|---|
| Python（系统级） | ❌ **未安装** | 只有 Microsoft Store 占位符 `WindowsApps\python.exe`，执行会跳转应用商店 |
| uv | ✅ 已安装 | `C:\Users\haku\.local\bin\uv.exe`，v0.11.3，已在 PATH |
| uv 自带 Python | ✅ | cpython-3.14.3（**版本过新**，torch / onnxruntime / pesq 等无 wheel） |
| Node.js | ✅ | v24.16.0 |
| npm | ✅ | 11.13.0 |
| git | ✅ | `C:\Users\haku\AppData\Local\Programs\Git\cmd\git.exe` |
| CMake | ❌ 未安装 | C++ 实时版（Phase 7）的前置条件 |
| MSVC Build Tools | ❌ 未找到 | 同上；仅存在残留的 `Visual Studio\COMMON` 空目录 |
| conda / miniconda | ❌ 未安装 | 不需要，用 uv 即可 |

### 由此产生的决策

1. **Python 版本锁定 3.11**。3.14 太新，PyTorch / onnxruntime / pesq / pystoi 在 Windows 上没有对应
   wheel，一旦源码编译就必然失败（且本机没有 MSVC）。由 uv 拉取独立的 3.11，不污染系统。
2. **全程 uv 管理**，不用 conda、不用系统 pip。命令统一为 `uv run ...`。
3. **Phase 7（C++ 实时版）延后**，并明确列出前置安装项，由用户确认后再装。前 6 个阶段完全不依赖它。

## 4. 关于"计算资源不足"的一点事实说明

本机 GPU 是 RTX 4000 Ada / 20 GB 显存，训练本项目规模的模型（0.05M ~ 4M 参数的流式增强网络）
在算力上是完全够的。真正卡住本地训练的是两点：

- **磁盘**：C: 只剩 18 GB，装不下任何有意义规模的训练数据；
- **数据获取**：DNS Challenge 的下载脚本按数百 GB 设计，网络与磁盘都不现实。

所以"**数据 + 训练在 Colab，其余在本地**"的分工是正确的，予以采纳。

相应地，训练代码会写成**位置无关**的：同一份 `src/rtse/` 包在 Colab 和本地跑同样的逻辑，
Colab notebook 只负责"下载数据 / 挂载 Drive / 调用训练入口"。将来若想改成本地小规模微调，
不需要改任何模型或训练代码。

## 5. 可复用的邻近资产

`C:\Users\haku\Documents\trae_projects\speech-processing-master\` 是一套语音处理教学仓库，
包含谱减法、维纳滤波、MMSE、DNN-IRM、SEGAN、FRCRN、Whisper 等实现与配套 PDF 讲义。

**定位：参考资料，不作为依赖。** 本项目所有算法自行实现，以保证代码风格统一、可控、可讲清楚。
遇到公式细节存疑时可以对照该仓库的讲义交叉验证。
