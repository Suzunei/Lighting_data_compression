<!-- LANG-SWITCH -->
**Language**: **简体中文** · [English](README.md)

> [!IMPORTANT]
> README 维护两份语言版本（[`README.md`](README.md) 主版本 · [`README.zh-CN.md`](README.zh-CN.md) 镜像），**任何改动须同时同步两份**。

---

# HI-NE-GBD：层级化神经高斯基分解

[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](HI-NE-GBD/runtime/cpp/)

> **用层级化的高斯引导神经解码器压缩预计算光照探针的球谐系数，并支持实时解压。**

一种基于高斯引导的 Moving Basis Decomposition (MBD) 层级方法，用于光照探针球谐 (SH) 系数的高效压缩与实时解压。

**参考文献**：
1. *Moving Basis Decomposition for Precomputed Light Transport*（EGSR 2021）
2. *Gaussian Compression for Precomputed Indirect Illumination*（SIGGRAPH 2025）

## 方法概述

HI-NE-GBD 采用 **Coarse-to-Fine（粗到细）层级解码架构**，把光照数据拆为低频全局光照分支与高频局部细节分支，在高压缩比下仍能保持高质量重建。

### 核心结构

```
f_final(x) = f_coarse(x) + f_fine(x) + 0.1 * residual(x)
```

**Coarse 分支（带 3D 高斯的 MBD）**：
- 用可学习的 3D 各向异性高斯核做空间插值
- Moving Basis Decomposition：`f_coarse(x) = Σ_l scale_l * c_l(x) * b_l(x)`
- 系数高斯（M）+ 基函数高斯（N）
- 每个高斯有完整协方差参数：position(3) + log_scale(3) + quaternion(4) + alpha(1)

**Fine 分支（高斯引导 MLP）**：
- Fine Gaussians (F)：稀疏锚点，自适应定位高频细节区域（边缘、阴影等）
- MLP 输入 = 位置编码(coords) + Fine 高斯权重（提供空间感知）
- 直接用高斯插值后的 features 作为补充细节源
- `f_fine(x) = MLP(PE(x), φ_fine(x)) + 0.2 * Σ φ_fine(x) * features`

**残差精修器**：
- 一个小 MLP，对融合后的输出做残差修正

### 训练策略（三阶段 Coarse-to-Fine）

| 阶段 | Epochs | 策略 | 说明 |
|------|--------|------|------|
| Stage 1 | 500 | Coarse Focus | 只更新 MBD 分支，λ_coarse 权重高 |
| Stage 2 | 2500 | Joint Training | 所有分支联合训练，λ_coarse 渐降（Curriculum Learning）|
| Stage 3 | 500 | Fine Focus | 聚焦 Fine 高斯 + MLP，λ_coarse 权重低 |

### 损失函数

```
L_total = L_final + λ_coarse * L_coarse + λ_reg * L_reg
```

- **L_final**：最终重建 MSE（支持按通道方差倒数加权）
- **L_coarse**：对 Coarse 分支的中间监督，鼓励 MBD 独立学到低频表征
- **L_reg**：正则项，防止尺度爆炸

## 数据格式

### 输入：ILCSampleData (.bin)

每个探针 32 个 float：
- 位置：3 floats (x, y, z)，世界坐标
- 半径：1 float
- SH 系数：27 floats (9 个 SH 阶次 × 3 RGB，RGB 交错存储)
  - 存储顺序：`[SH0_R, SH0_G, SH0_B, SH1_R, SH1_G, SH1_B, ..., SH8_R, SH8_G, SH8_B]`
- Shadow：1 float

### 输出：压缩模型 (.pth)

包含全部可学习参数（FP16 量化存储），支持任意坐标的实时 SH 系数解压。

## 项目结构

```
.
├── docs/
│   └── architecture.png          # 架构图（项目级文档）
├── HI-NE-GBD/                    # 生产管线（用这套）
│   ├── buildtime/
│   │   ├── HI-NE-GBD.py          #   主训练脚本（离线压缩）
│   │   └── probe_reader.py       #   探针读取工具
│   ├── runtime/
│   │   ├── decoder.py            #   实时解压器（Python）
│   │   ├── export_model.py       #   PyTorch -> UCommon .uasset 导出器
│   │   └── cpp/                  #   C++ 运行时（MSVC）
│   │       ├── HINEGBD.h/cpp     #     解码库
│   │       ├── main.cpp          #     demo 入口
│   │       └── CMakeLists.txt
│   └── probedata/
│       └── ILCSampleData_0.bin   #   光照探针原始数据（仓库唯一一份）
├── experiments/
│   ├── ablation/                 # 组件消融脚本（多数跑合成信号）
│   │   ├── HI-NE-GBD.py          #   真实探针重训（不保存模型，只可视化）
│   │   ├── MBD_gaussian.py       #   去掉 MLP（合成信号）
│   │   ├── MBD_gaussian_MLP.py   #   非层级版（合成信号）
│   │   └── gaussian_MLP.py       #   去掉 MBD（合成信号）
│   └── legacy/                   # 早期探索脚本（仅合成信号）
└── output/                       # 历史运行的图片
```

`compressed_model.pth`（训练产出）和 `hinegbd_model.uasset`（导出产出）不入库——通过运行 buildtime / export 脚本产生。

## 使用方法

脚本路径都基于 `__file__`，从任意工作目录调用都能跑。

### 1. 训练（离线压缩）

```bash
python HI-NE-GBD/buildtime/HI-NE-GBD.py
```

在 `HI-NE-GBD/probedata/ILCSampleData_0.bin` 上跑 3500 epochs（500 coarse + 2500 joint + 500 fine），写出 `HI-NE-GBD/runtime/compressed_model.pth`（FP16 量化）。

### 2. 实时解压（Python）

```python
from decoder import HINEGBDDecoder

decoder = HINEGBDDecoder("compressed_model.pth")

# 单点查询（世界坐标）
sh_coeffs = decoder.decode(x=100.0, y=50.0, z=-200.0)

# 批量查询
import numpy as np
coords = np.array([[100, 50, -200], [110, 60, -180]], dtype=np.float32)
sh_batch = decoder.decode_batch(coords)
```

### 3. 交互式 / Benchmark / 单点查询命令行

```bash
python HI-NE-GBD/runtime/decoder.py --interactive
python HI-NE-GBD/runtime/decoder.py --benchmark
python HI-NE-GBD/runtime/decoder.py --coords 100 50 -200            # 世界坐标
python HI-NE-GBD/runtime/decoder.py --coords 0.5 0.5 0.5 --normalized
```

`--device {cpu,cuda}` 覆盖自动检测；`--model PATH` 覆盖默认 `compressed_model.pth`。

### 4. 导出到 C++ 运行时

```bash
python HI-NE-GBD/runtime/export_model.py \
  --model compressed_model.pth --output hinegbd_model.uasset --verify
```

接着构建 C++ demo（Windows / MSVC）：

```bash
cmake -S HI-NE-GBD/runtime/cpp -B HI-NE-GBD/runtime/cpp/build
cmake --build HI-NE-GBD/runtime/cpp/build --config Release
HI-NE-GBD/runtime/cpp/build/bin/HINEGBD_Demo.exe hinegbd_model.uasset
```

## 模型配置（默认参数）

| 参数 | 值 | 说明 |
|------|------|------|
| `num_bases` (L) | 16 | MBD 基函数数量 |
| `coeff_res` (M) | 64 | 系数高斯数量 |
| `basis_res` (N) | 64 | 基函数高斯数量 |
| `data_dim` (D) | 27 | 数据维度（SH 系数）|
| `fine_gaussian_res` (F) | 32 | Fine 高斯锚点数量 |
| `mlp_hidden` | 128 | MLP 隐藏层大小 |
| `pe_num_freqs` | 6 | 位置编码频率数 |
| `fine_mlp_depth` | 3 | Fine MLP 深度 |
| `fine_kernel_scale` | 0.05 | Fine 高斯初始尺度 |

## 关键创新点

1. **层级解码**：MBD 处理低频 + 高斯+MLP 处理高频
2. **Fine Gaussians**：稀疏锚点自适应学习高频区域（边缘、阴影过渡）
3. **高斯引导 MLP**：高斯权重提供空间感知，引导 MLP 学习局部细节
4. **MBD 可学习 Scale**：每个基函数有独立可学习的 scale 因子
5. **位置编码**：正弦位置编码增强高频学习能力
6. **加性残差融合**：无 gate 网络开销（coarse + fine 直接相加）
7. **中间监督**：对 Coarse 分支独立监督
8. **课程学习**：λ_coarse 在训练过程中渐降
9. **按通道方差倒数加权**：均衡各 SH 阶次的梯度
10. **FP16 后训练量化**：训练后做 Float16 量化，压缩比加倍

## 评估指标

- **PSNR**（峰值信噪比）：按通道 PSNR 取均值
- **SSIM**（结构相似度）：基于 Morton Z 序空间排序的滑窗 SSIM
- **压缩比**：原始数据大小 / 压缩模型大小
- **按 SH 阶次拆分**：分别评估各阶 SH（L0/L1/L2）

## 依赖

- Python 3.8+
- PyTorch >= 1.10
- NumPy
- SciPy
- Matplotlib

## 消融研究

`experiments/ablation/` 目录下含组件消融脚本（除标注外均跑 `test_signal_3d.py` 的合成信号）：
- `MBD_gaussian.py`：MBD + 高斯（去 MLP）
- `MBD_gaussian_MLP.py`：MBD + 高斯 + MLP（非层级）
- `gaussian_MLP.py`：高斯 + MLP（去 MBD）
- `HI-NE-GBD.py`：层级版完整模型，跑真实探针数据 (`ILCSampleData_0.bin`)

`experiments/legacy/` 留存早期探索脚本（`opt.py`、`HI_TEST.py`、`MBD_*.py`、`gaussian_control.py`），作历史参考。
