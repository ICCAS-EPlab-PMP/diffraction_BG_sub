# BGsub — X射线衍射数据背景扣除套件

# BGsub — Background Subtraction Suite for X-ray Diffraction Data

---

## 简介 / Introduction

BGsub 是一款基于 [silx](https://silx.org/) GUI 的 X 射线衍射（SAXS/WAXS/XRD）数据背景扣除桌面应用。支持**有参考背景扣除**和**无参考数学拟合**两种工作模式，可处理 TIFF、EDF、HDF5 等格式的探测器图像，并集成 SSRF 电离室透射率自动校正功能。

BGsub is a desktop application for background subtraction of X-ray diffraction (SAXS/WAXS/XRD) data, built on the [silx](https://silx.org/) GUI framework. It supports both **reference-based background subtraction** and **mathematical fitting without reference**, handles detector images in TIFF, EDF, and HDF5 formats, and integrates SSRF ionchamber transmission auto-correction.

**主要目标用户：中国同步辐射用户 / Primary target users: Chinese synchrotron radiation users**
<img width="835" height="946" alt="SubBG" src="https://github.com/user-attachments/assets/f27b875b-58e7-4428-b22a-9b98c77aa4f7" />

---

## 功能特点 / Features

### 两种工作模式 / Two Working Modes

| 模式 | 说明 |
|------|------|
| **有参考背景扣除** (Reference-based) | 使用已知背景图像进行信号扣除：`result = sample / T - background` |
| **无参考数学拟合** (Mathematical fitting) | 使用 Rolling Ball、多项式拟合、形态学操作等算法估计背景 |

### 支持的背景估计算法 / Supported Background Estimation Methods

- **Rolling Ball** (经典/平滑) / Rolling Ball (classic/smooth)
- **多项式拟合** / Polynomial fitting
- **形态学操作** / Morphological operations (opening/closing)

### 支持的数据格式 / Supported Data Formats || (通过 [fabio](https://fabio.readthedocs.io/))

- TIFF / EDF 
- HDF5 (.h5 / .hdf5) — Eiger 等探测器
- 多帧堆叠 / Multi-frame stacks
- SSRF 电离室文件 (.Ionchamber / .txt)

### GUI 功能 / GUI Features

- **CompareImages** — 图像对比查看 (silx)
- **ImageView** — 带直方图的图像显示 (silx)
- **StackView** — 多帧浏览 (silx)
- 批量处理进度条 / Batch processing progress bar
- 日志窗口 / Log window

---

## 安装 / Installation

### 方式一：pip 安装（推荐）

```bash
pip install -e .
```

### 方式二：直接运行

```bash
python run_app.py
```

### 依赖 / Requirements

- Python >= 3.8
- numpy >= 1.20
- scipy >= 1.7
- fabio >= 0.12
- h5py >= 3.0
- silx >= 1.0
- pandas >= 1.3

### 预打包下载 / Pre-built Distribution

无需安装 Python 环境，直接下载解压即可使用：

**下载链接 / Download: [http://www.polymcrystal.com/download/Xray_diffraction_BG_sub.zip](http://www.polymcrystal.com/download/Xray_diffraction_BG_sub.zip)**

---

## 快速开始 / Quick Start

### 启动 2D 程序 / Launch 2D Application

```bash
# 方式一
python run_app.py

# 方式二（安装后）
BGsub

# 方式三
python -m BGsub.gui.main_window
```

### 启动 1D 程序 / Launch 1D Application

```bash
# 方式一
python run_1d.py

# 方式二（安装后）
BGsub-1d

# 方式三
python -m BGsub.standalone_1d.main
```

### 有参考背景扣除流程 / Reference-based Background Subtraction

1. **文件选择** — 点击「📂 选择文件」或「📁 选择文件夹」加载 TIFF/EDF 图像
2. **选择背景** — 在下拉框中选择背景文件（或加载独立背景）
3. **设置透过率** — 手动输入统一值，或选择「电离室文件」自动计算
4. **执行处理** — 点击「▶ 开始处理」
5. **导出结果** — 选择格式（EDF/TIFF/HDF5）并保存

### 无参考背景拟合流程 / Mathematical Fitting Workflow

1. 加载图像后，选择「数学拟合」模式
2. 选择算法（Rolling Ball / 多项式 / 形态学）
3. 设置参数（半径、阶数等）
4. 执行并预览结果

---

## 1D 独立程序使用指南 / Standalone 1D Application Guide

### 三种处理模式 / Three Processing Modes

| 模式 | 适用场景 | 核心算法 |
|------|----------|----------|
| **T-背景扣除** | 传统样品曲线 / T 修正背景曲线扣除 | `result = sample / (T/100) - background` |
| **1D 形态学** | 对已有 1D 曲线使用形态学操作估计背景 | 结构元素膨胀/腐蚀 (morphological opening) |
| **1D 拟合** | 对已有 1D 曲线使用多项式或滚球算法拟合背景 | 多项式拟合 / 滚球算法 (rolling ball) |

### 支持的输入格式 / Supported Input Formats

| 格式 | 扩展名 |
|------|--------|
| **XY / DAT / TXT** | `.xy`, `.dat`, `.txt` (支持 single XY / XYXY / XYYY) |
| **CSV** | `.csv` (支持 single XY / XYYY) |
| **GR** | `.gr` |

### 真实样例验证路径 / Real Sample Validation Paths

项目包含真实样例数据，用于验证 1D 处理流程：

#### BGsub/test/ 本地测试样例
```
BGsub/test/wkf-G1-Q-30-2-waxs-11.9s_00001.tif     # 样品图像
BGsub/test/wkf-G1-Q-30-2-waxs-11.9s_00002.tif     # 样品图像
BGsub/test/wkf-G1-W-30-1-BG-waxs_00001.tif        # 背景图像
BGsub/test/wkf-G1-Q-30-2-11.9s_1.Ionchamber       # 样品电离室文件
BGsub/test/wkf-G1-W-30-1-BG.Ionchamber            # 背景电离室文件
```

### 内存策略与性能优化 / Memory Strategy & Performance

1. **逐文件流式处理**
   - 处理大文件时逐个加载，避免同时加载所有文件到内存
   - 每处理完一个文件立即释放内存
   - 支持中断处理，用户可随时停止

2. **预览限制**
   - 批量处理时仅自动预览前 N 个结果（由界面中的 Preview limit 控制）
   - 避免生成大量中间图像占用内存
   - 用户可单独选择文件进行详细预览

3. **进度反馈**
   - 实时显示处理进度和剩余时间估算
   - 处理过程中显示当前文件信息
   - 错误发生时提供详细错误信息，不影响其他文件处理

### 导出规则 / Export Rules

1. **文件命名**
   ```
   原始文件: sample_001.tif
   导出文件: sample_001_raw.csv
   导出文件: sample_001_bg.csv (可选)
   导出文件: sample_001_sub.csv (可选)
   ```

2. **格式支持**
   - **XY / TXT / GR**: 两列文本格式
   - **CSV**: 两列逗号分隔格式
   - **NPY**: `N x 2` 的 NumPy 数组，第一列为 x，第二列为所选 y 数据
   - **HDF5**: 支持单曲线或多曲线合并导出
   - **预览图像导出**: 当前界面支持导出 PNG / SVG / PDF 图像

3. **元数据保留**
   - 文本格式会写入 source / process_mode / data_type 等头信息
   - 记录处理参数（模式、算法、参数值）
   - NPY 仅保存数值数组，不内嵌文本元数据

### 使用流程示例 / Usage Example

1. **1D 曲线背景扣除流程 / 1D Curve Background Subtraction**
   ```
   加载 1D 曲线 → 选择处理模式（形态学/拟合）→ 设置参数 → 预览效果 → 导出
   Load 1D curves → Select mode (morphology/fitting) → Set parameters → Preview → Export
   ```

2. **批量处理流程 / Batch Processing**
   ```
   选择文件夹 → 设置处理参数 → 批量处理 → 统一导出到指定目录
   Select folder → Set parameters → Batch process → Export all to output directory
   ```

3. **多列文本流程 / Multi-column Text Workflow**
   ```
   加载多列文本 → 指认列结构（single / XYXY / XYYY）→ 处理 → 选择逐文件或合并单文件导出
   Load multi-column text → choose layout (single / XYXY / XYYY) → process → export per-file or merged
   ```

---

## 项目结构 / Project Structure

```
Xray_diffraction_BG_sub/
├── BGsub/                          # 核心包 / Core package
│   ├── __init__.py
│   ├── _version.py
│   ├── core/                       # 核心算法 / Core algorithms
│   │   ├── bg_fitter.py            # 背景拟合接口 / Background fitting interface
│   │   ├── bg_math.py              # 背景估计算法 / Background estimation algorithms
│   │   ├── curve_data.py           # 1D 曲线数据模型 / 1D curve data model
│   │   ├── curve_processor.py      # 1D 曲线处理器 / 1D curve processor
│   │   └── task_pipeline.py        # 任务流水线 / Task pipeline
│   ├── io/                         # 文件输入输出 / File I/O
│   │   ├── curve_io.py             # 1D 曲线读写 / 1D curve I/O
│   │   └── image_io.py             # TIFF/EDF/H5 读写 / 2D image I/O
│   ├── gui/                        # 2D 图形界面 / 2D GUI
│   │   ├── main_window.py          # 2D 主窗口 / Main window (TiffTab / H5Tab)
│   │   ├── compare_widget.py       # CompareImages 封装 / CompareImages wrapper
│   │   └── stack_viewer.py         # StackView 封装 / StackView wrapper
│   ├── standalone_1d/              # 独立 1D 程序 / Standalone 1D application
│   │   ├── main.py                 # 1D 程序入口 / Entry point
│   │   ├── curve_window.py         # 1D 主窗口 / Main window
│   │   ├── mode_panels.py          # 1D 模式面板 / Mode panels
│   │   └── plot_canvas.py          # 1D 预览画布 / Preview canvas
│   └── utils/
│       └── common.py               # 通用工具 / Common utilities
├── BGsub_skills/                   # OpenCode Skill（中英双语 / Bilingual CN/EN）
│   ├── SKILL.md                    # Skill 定义文件 / Skill definition
│   ├── scripts/                    # 可独立运行的脚本 / Standalone scripts
│   │   ├── bg_subtract.py          # 2D 单文件背景扣除 / 2D single-file subtraction
│   │   ├── bg_batch.py             # 2D 批量背景扣除 / 2D batch subtraction
│   │   ├── ionchamber.py           # 电离室分析 / Ionchamber analysis
│   │   ├── format_convert.py       # 格式转换 / Format conversion
│   │   ├── compare_images.py       # silx 图像对比 / silx CompareImages
│   │   ├── curve_process_1d.py     # 1D 单曲线处理 / 1D single-curve processing
│   │   └── curve_batch_1d.py       # 1D 批量曲线处理 / 1D batch-curve processing
│   └── references/                 # 参考文档 / Reference docs
│       ├── quickstart.md           # 快速入门 / Quick start guide
│       ├── cli_commands.md         # 命令参考 / Command reference
│       └── examples.md             # 使用示例 / Usage examples
├── logo.ico                        # 应用图标 / Application icon
├── requirements.txt                # Python 依赖 / Dependencies
├── setup.py                        # 安装脚本 / Install script
├── run_app.py                      # 2D 启动脚本 / 2D launcher
├── run_1d.py                       # 1D 启动脚本 / 1D launcher
└── README.md
```

---

## Skill 使用方法 / Skill Usage

本项目包含 `BGsub_skills/`（中英双语 / Bilingual）Skill，可供 [OpenCode](https://opencode.ai/) AI 编码助手直接调用。所有脚本均为**完整可执行代码**，无需额外生成即可运行。

This project includes a bilingual `BGsub_skills/` Skill for [OpenCode](https://opencode.ai/) AI coding assistants. All scripts are **complete, executable code** — no additional generation needed.

### Skill 核心能力 / Skill Core Capabilities

| # | 能力 / Capability | 脚本 / Script | 适用场景 / Use Case |
|---|---|---|---|
| 1 | 2D 单文件背景扣除 / 2D single-file subtraction | `bg_subtract.py` | 使用参考背景扣除单张图像 |
| 2 | 2D 批量背景扣除 / 2D batch subtraction | `bg_batch.py` | 整个目录批量背景扣除 |
| 3 | 电离室透射率分析 / Ionchamber analysis | `ionchamber.py` | 读取 .Ionchamber 文件计算透射率 |
| 4 | 格式转换 / Format conversion | `format_convert.py` | TIFF / EDF / H5 格式互转 |
| 5 | silx 图像对比 / silx CompareImages | `compare_images.py` | 弹窗对比两张图像差异 |
| 6 | 1D 单曲线处理 / 1D single-curve processing | `curve_process_1d.py` | 对单条 1D 曲线进行背景估计与扣除 |
| 7 | 1D 批量曲线处理 / 1D batch-curve processing | `curve_batch_1d.py` | 批量处理 1D 曲线（形态学/拟合/滚球） |

### Skill 使用示例 / Skill Usage Examples

在 OpenCode 中加载 Skill 后，直接描述需求即可触发：

```
# 2D 背景扣除示例 / 2D subtraction example
用户：帮我用 background.tif 扣除 sample.tif 的背景，透射率 95%
→ Skill 自动执行 bg_subtract.py

# 1D 曲线处理示例 / 1D curve processing example
用户：对 data.xy 做形态学背景扣除，半径 50
→ Skill 自动执行 curve_process_1d.py --method morph --radius 50

# 批量处理示例 / Batch processing example
用户：批量处理 ./curves/ 目录下所有 .xy 文件，用多项式拟合，4阶
→ Skill 自动执行 curve_batch_1d.py --method poly --degree 4
```

### 脚本独立运行 / Standalone Script Usage

所有脚本均可直接在命令行运行，无需 OpenCode 环境：

All scripts can be run directly from the command line without OpenCode:

```bash
# 2D 单文件背景扣除 / 2D single-file subtraction
python BGsub_skills/scripts/bg_subtract.py sample.tif background.tif --ion-dir ./ionchamber -o result.tif

# 2D 批量处理 / 2D batch processing
python BGsub_skills/scripts/bg_batch.py ./data background.tif --ion-dir ./ionchamber -o ./results

# 电离室数据分析 / Ionchamber analysis
python BGsub_skills/scripts/ionchamber.py sample.Ionchamber

# 格式转换 / Format conversion
python BGsub_skills/scripts/format_convert.py input.tif -o output.edf --format edf

# silx 图像对比（弹窗）/ silx CompareImages (popup)
python BGsub_skills/scripts/compare_images.py sample.tif background.tif

# 1D 单曲线处理 / 1D single-curve processing
python BGsub_skills/scripts/curve_process_1d.py data.xy --method morph --radius 50 -o result.xy

# 1D 批量曲线处理 / 1D batch-curve processing
python BGsub_skills/scripts/curve_batch_1d.py ./curves/ --method poly --degree 4 -o ./output/
```

更多示例详见 `BGsub_skills/references/examples.md`。

For more examples, see `BGsub_skills/references/examples.md`.

---

## SSRF 电离室文件格式 / SSRF Ionchamber File Format

电离室文件为纯文本格式（扩展名 `.Ionchamber` 或 `.txt`）：

```
# Time  Ionchamber0  Ionchamber1  Ionchamber2
2026-01-19 12:34:34.130671238  2.807135e-7  2.388761e-8  9.71103e-10
```

| 通道 | 说明 |
|------|------|
| Ionchamber0 | 入射光强度 |
| Ionchamber1 | 透射光（中间） |
| Ionchamber2 | 透射光（主测量） |

### 透射率计算 / Transmission Calculation

```
T = I_sample / I_background × 100%
result = sample / T - background
```

---

## 未来计划 / TODO

- [ ] **更小的内存占用** — 优化大图像处理，降低内存峰值 / Smaller memory footprint — optimize large image processing
- [ ] **更顺滑的界面** — 提升 GUI 响应速度，改善用户体验 / Smoother UI — improve responsiveness
- [ ] **中英文切换** — 实现界面语言动态切换 / Chinese/English toggle — dynamic language switching
- [ ] **一维曲线处理验证** - Verify 1D curve function

### 长期更新目标 / Long-term Goals

- [ ] **数学方式背景估计** — 完善 Rolling Ball、多项式拟合、形态学操作的参数自动化 / Mathematical background estimation — improve parameter automation for Rolling Ball, polynomial fitting, morphological operations
- [ ] **全局优化方式** — 引入全局优化算法（如 Nelder-Mead、BFGS）进行几何参数和背景的联合精修 / Global optimization — joint refinement of geometry parameters and background using global optimization algorithms (Nelder-Mead, BFGS, etc.)

---

## 致谢 / Acknowledgments

本工具的开发得到了以下开源项目的支持：

Development of this tool was made possible by the following open-source projects:

### silx

> Sébastien Range, Pierre paleo, Frédéric-Emmanuel Picca, Tim P. A. H. M. Rijks, Vernon D. P. T. N. G. C. L. van der Schors, et al. (2025). **silx: Python GUI library for X-ray imaging and data analysis**. Zenodo. [https://doi.org/10.5281/zenodo.591709](https://doi.org/10.5281/zenodo.591709)

### fabio

> Knudsen, E. B., Sørensen, H. O., Wright, J. P., Goret, G. & Kieffer, J. (2013). **fabio: Processing of 2D X-ray diffraction data**. J. Appl. Cryst. 46, 537-539. [http://dx.doi.org/10.1107/S0021889813000150](http://dx.doi.org/10.1107/S0021889813000150)

### Qt / PySide

> The Qt Company. **Qt: A cross-platform UI and application framework**. [https://www.qt.io/](https://www.qt.io/)
> PySide6 is the official Python bindings for Qt, providing access to the complete Qt API. [https://doc.qt.io/qtforpython/](https://doc.qt.io/qtforpython/)

---

## 许可证 / License

MIT License

---

## 联系方式 / Contact

如有问题或建议，请通过 GitHub Issues 联系。

For questions or suggestions, please contact via GitHub Issues.

**下载链接 / Download: [http://www.polymcrystal.com/download/Xray_diffraction_BG_sub.zip](http://www.polymcrystal.com/download/Xray_diffraction_BG_sub.zip)**
