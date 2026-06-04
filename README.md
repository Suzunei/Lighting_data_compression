Used moving gaussian decomposition and mlp to compress the pre-computed light transport data - light-probe  
Reference:   
1.Moving Basis Decomposition for Precomputed Light Transport（EGSR 2021）  
2.Gaussian Compression for Precomputed Indirect Illumination（SIGGRAPH 2025）  

# HI-NE-GBD: Hierarchical Neural-Enhanced Gaussian Basis Decomposition

A hierarchical Gaussian-guided Moving Basis Decomposition (MBD) method for efficient compression and real-time decompression of Spherical Harmonics (SH) coefficients from light probes.

## Method Overview

HI-NE-GBD employs a **Coarse-to-Fine hierarchical decoding architecture** that decomposes lighting data into a low-frequency global illumination branch and a high-frequency local detail branch, achieving high-quality reconstruction under high compression ratios.

### Core Architecture

```
f_final(x) = f_coarse(x) + f_fine(x) + 0.1 * residual(x)
```

**Coarse Branch (MBD with 3D Gaussians)**:
- Uses learnable 3D anisotropic Gaussian kernels for spatial interpolation
- Moving Basis Decomposition: `f_coarse(x) = Σ_l scale_l * c_l(x) * b_l(x)`
- Coefficient Gaussians (M) + Basis Gaussians (N)
- Each Gaussian has full covariance parameters: position(3) + log_scale(3) + quaternion(4) + alpha(1)

**Fine Branch (Gaussian-Guided MLP)**:
- Fine Gaussians (F): Sparse anchors that learn to locate high-frequency detail regions (edges, shadows, etc.)
- MLP input = Positional Encoding(coords) + Fine Gaussian weights (spatial awareness)
- Direct Gaussian-interpolated features as supplementary detail source
- `f_fine(x) = MLP(PE(x), φ_fine(x)) + 0.2 * Σ φ_fine(x) * features`

**Residual Refiner**:
- A small MLP that performs residual correction on the blended output

### Training Strategy (Three-Stage Coarse-to-Fine)

| Stage | Epochs | Strategy | Description |
|-------|--------|----------|-------------|
| Stage 1 | 500 | Coarse Focus | Only update MBD branch, high λ_coarse weight |
| Stage 2 | 2500 | Joint Training | Train all branches jointly, λ_coarse decays gradually (Curriculum Learning) |
| Stage 3 | 500 | Fine Focus | Focus on Fine Gaussians + MLP, low λ_coarse weight |

### Loss Function

```
L_total = L_final + λ_coarse * L_coarse + λ_reg * L_reg
```

- **L_final**: Final reconstruction MSE (supports per-channel variance-inverse weighting)
- **L_coarse**: Intermediate supervision on the coarse branch, encouraging MBD to independently learn low-frequency representation
- **L_reg**: Regularization term to prevent scale explosion

## Data Format

### Input: ILCSampleData (.bin)

Each probe consists of 32 floats:
- Position: 3 floats (x, y, z) — world coordinates
- Radius: 1 float
- SH Coefficients: 27 floats (9 SH bands × 3 RGB, RGB-interleaved)
  - Storage order: `[SH0_R, SH0_G, SH0_B, SH1_R, SH1_G, SH1_B, ..., SH8_R, SH8_G, SH8_B]`
- Shadow: 1 float

### Output: Compressed Model (.pth)

Contains all learnable parameters (FP16 quantized storage), supporting real-time decompression of SH coefficients at arbitrary coordinates.

## Project Structure

```
HI-NE-GBD/
├── buildtime/
│   ├── HI-NE-GBD.py          # Main training script (offline compression)
│   ├── probe_reader.py        # Probe data reader utility
│   └── HI-NE-GBD.png          # Training results visualization
├── runtime/
│   ├── decoder.py             # Real-time decoder module (Python)
│   ├── export_model.py        # Model export utility
│   ├── compressed_model.pth   # Compressed model file
│   └── cpp/                   # C++ runtime implementation
│       ├── HINEGBD.h/cpp      # C++ decoder
│       ├── main.cpp           # Test entry point
│       └── CMakeLists.txt     # Build configuration
├── probedata/
│   └── ILCSampleData_0.bin    # Light probe raw data
└── HI-NE-GBD.png              # Architecture diagram
```

## Usage

### 1. Training (Offline Compression)

```powershell
cd HI-NE-GBD/buildtime
python HI-NE-GBD.py
```

After training completes, the compressed model is automatically saved to `runtime/compressed_model.pth`.

### 2. Real-time Decompression (Python)

```python
from decoder import HINEGBDDecoder

# Load model
decoder = HINEGBDDecoder("compressed_model.pth")

# Single point query (world coordinates)
sh_coeffs = decoder.decode(x=100.0, y=50.0, z=-200.0)

# Batch query
import numpy as np
coords = np.array([[100, 50, -200], [110, 60, -180]], dtype=np.float32)
sh_batch = decoder.decode_batch(coords)
```

### 3. Interactive Command-Line Mode

```powershell
cd HI-NE-GBD/runtime
python decoder.py --model compressed_model.pth --interactive
```

### 4. Performance Benchmark

```powershell
python decoder.py --model compressed_model.pth --benchmark
```

## Model Configuration (Default Parameters)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_bases` (L) | 16 | Number of MBD bases |
| `coeff_res` (M) | 64 | Number of coefficient Gaussians |
| `basis_res` (N) | 64 | Number of basis Gaussians |
| `data_dim` (D) | 27 | Data dimension (SH coefficients) |
| `fine_gaussian_res` (F) | 32 | Number of fine Gaussian anchors |
| `mlp_hidden` | 128 | MLP hidden layer size |
| `pe_num_freqs` | 6 | Number of positional encoding frequencies |
| `fine_mlp_depth` | 3 | Fine MLP depth |
| `fine_kernel_scale` | 0.05 | Fine Gaussian initial scale |

## Key Innovations

1. **Hierarchical Decoding**: MBD for low-frequency + Gaussian+MLP for high-frequency
2. **Fine Gaussians**: Sparse anchors that adaptively learn high-frequency regions (edges, shadow transitions)
3. **Gaussian-Guided MLP**: Gaussian weights provide spatial awareness, guiding MLP to learn local details
4. **MBD Learnable Scale**: Per-basis learnable scale factors
5. **Positional Encoding**: Sinusoidal encoding to enhance high-frequency learning
6. **Additive Residual Blending**: No gate network overhead (coarse + fine)
7. **Intermediate Supervision**: Independent supervision on the coarse branch
8. **Curriculum Learning**: λ_coarse decays gradually during training
9. **Per-Channel Variance-Inverse Weighting**: Balances gradients across SH orders
10. **FP16 Post-Training Quantization**: Float16 quantization after training, doubling compression ratio

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Per-channel PSNR averaged
- **SSIM** (Structural Similarity): Sliding-window SSIM based on Morton Z-order spatial sorting
- **Compression Ratio**: Original data size / compressed model size
- **Per-SH-Order Breakdown**: Separate evaluation for each SH order (L0/L1/L2)

## Dependencies

- Python 3.8+
- PyTorch >= 1.10
- NumPy
- SciPy
- Matplotlib

## Ablation Studies

The `src/` directory contains ablation experiment scripts for individual components:
- `MBD_gaussian.py`: MBD + Gaussian (no MLP)
- `MBD_gaussian_MLP.py`: MBD + Gaussian + MLP
- `gaussian_MLP.py`: Gaussian + MLP (no MBD)
- `HI-NE-ablation.py`: HI-NE-GBD ablation comparison

