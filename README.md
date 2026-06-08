<!-- LANG-SWITCH -->
**Language**: **English** · [简体中文](README.zh-CN.md)

> [!IMPORTANT]
> README is maintained in two languages ([`README.md`](README.md) canonical · [`README.zh-CN.md`](README.zh-CN.md) mirror). **Any change must update both in the same commit.**

---

# HI-NE-GBD: Hierarchical Neural Gaussian Basis Decomposition

[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](HI-NE-GBD/runtime/cpp/)

> **Compress pre-computed light-probe SH coefficients with a hierarchical Gaussian-guided neural decoder, decompress in real time.**

A hierarchical Gaussian-guided Moving Basis Decomposition (MBD) method for efficient compression and real-time decompression of Spherical Harmonics (SH) coefficients from light probes.

**References**:
1. *Moving Basis Decomposition for Precomputed Light Transport* (EGSR 2021)
2. *Gaussian Compression for Precomputed Indirect Illumination* (SIGGRAPH 2025)

## Method Overview

HI-NE-GBD employs a **Coarse-to-Fine hierarchical decoding architecture** that decomposes lighting data into a low-frequency global illumination branch and a high-frequency local detail branch, achieving high-quality reconstruction under high compression ratios.

### Core Equation

$$
f_{\text{final}}(\mathbf{x}) = \underbrace{f_{\text{coarse}}(\mathbf{x})}_{\text{MBD low-freq}} + \underbrace{f_{\text{fine}}(\mathbf{x})}_{\text{Gaussian-guided MLP}} + 0.1 \cdot \underbrace{r(\mathbf{x})}_{\text{27-dim refiner}}
$$

### Data Flow & Channel Dimensions

The diagram below traces a query of $B$ probes through the network. **Edge labels are tensor shapes** — every channel-count transition is annotated.

```mermaid
flowchart TD
    INPUT["coords<br/><b>[B, 3]</b>"]:::input

    INPUT --> PE["Positional Encoding<br/>3 + 3·2·6 freqs"]:::op
    INPUT --> COEFF_G["Coeff Gaussians<br/>M = 64 anchors"]:::op
    INPUT --> BASIS_G["Basis Gaussians<br/>N = 64 anchors"]:::op
    INPUT --> FINE_G["Fine Gaussians<br/>F = 32 anchors"]:::op

    COEFF_G -->|"φ<br/>[B, 64]"| MC["× C<br/>[64, 16]"]:::op
    BASIS_G -->|"ψ<br/>[B, 64]"| MB["× B<br/>[64, 16, 27]"]:::op
    MC -->|"c_l(x)<br/>[B, 16]"| MBD["MBD reduce<br/>Σ_l scale_l · c_l · b_l"]:::op
    MB -->|"b_l(x)<br/>[B, 16, 27]"| MBD
    MBD -->|"f_coarse<br/><b>[B, 27]</b>"| BLEND(("+"))

    FINE_G -->|"φ_fine<br/>[B, 32]"| MIX["× V (latent)<br/>[F=32, K=32]<br/>↓ scene content enters here"]:::op
    PE -->|"[B, 39]"| CAT["concat<br/>[B, 71]"]:::op
    MIX -->|"mixed latent F(x)<br/><b>[B, 32]</b>"| CAT
    CAT --> FMLP["fine_mlp<br/>71→128→128→128→27<br/>(GPC Eq.7)"]:::mlp
    FMLP -->|"f_fine<br/><b>[B, 27]</b>"| BLEND

    BLEND -->|"blended<br/>[B, 27]"| REFCAT["concat with coords"]:::op
    INPUT -.->|"[B, 3]"| REFCAT
    REFCAT -->|"[B, 30]"| REF["residual_refiner<br/>30→64→27"]:::mlp
    BLEND -->|"[B, 27]"| FINAL(("+"))
    REF -->|"[B, 27]<br/>× 0.1"| FINAL
    FINAL -->|"<b>[B, 27]</b>"| OUT["SH coefficients<br/>9 bands × RGB"]:::output

    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef output fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    classDef op fill:#fff9c4,stroke:#f57f17
    classDef mlp fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

> [!NOTE]
> Three independent dimensions co-exist:
> **anchor count** (M=64, N=64, F=32) — how many Gaussian kernels per branch ·
> **basis count** (L=16) — MBD's rank-like factorization width ·
> **data dim** (D=27) — 9 SH bands × 3 RGB channels (interleaved).

### Branch Details

#### Coarse Branch — MBD with 3D Gaussians

$$
f_{\text{coarse}}(\mathbf{x}) = \sum_{l=1}^{L} s_l \cdot c_l(\mathbf{x}) \cdot \mathbf{b}_l(\mathbf{x}), \quad
c_l(\mathbf{x}) = \sum_{m=1}^{M} \varphi_m(\mathbf{x})\, C_{m,l}, \quad
\mathbf{b}_l(\mathbf{x}) = \sum_{n=1}^{N} \psi_n(\mathbf{x})\, \mathbf{B}_{n,l}
$$

Each Gaussian carries full covariance parameters: `position(3) + log_scale(3) + quaternion(4) + alpha(1)` — anisotropic, rotatable, intensity-weighted.

#### Fine Branch — Mixed-Latent MLP (paper-aligned, GPC Eq.5/7)

The fine branch follows *Gaussian Compression for Precomputed Indirect Illumination* (SIGGRAPH 2025): each fine anchor carries a $K$-dim **latent** $\mathbf{V}_j$, the Gaussian weights mix those latents into a per-query feature $F(\mathbf{x})$, and the MLP maps that — concatenated with positional encoding — to the $D$-dim output:

$$
\underbrace{\boldsymbol{\varphi}(\mathbf{x}) \in \mathbb{R}^F}_{\text{Eq.6 weights}}, \quad
\underbrace{F(\mathbf{x}) = \boldsymbol{\varphi}(\mathbf{x})\,\mathbf{V}}_{\text{Eq.5: }[B, K]}, \quad
\underbrace{f_{\text{fine}}(\mathbf{x}) = \text{MLP}_\theta\!\Big(\text{PE}(\mathbf{x}) \,\Vert\, F(\mathbf{x})\Big)}_{\text{Eq.7: }[B, D]}
$$

with $F=32$ anchors, $K=32$ latent width, and $\mathbf{V} \in \mathbb{R}^{32 \times 32}$. The MLP's first layer is $71{\to}128$ (PE 39 + latent 32) and the output layer is zero-initialized so the fine branch starts at zero and learns a residual on top of MBD.

```mermaid
flowchart TD
    INPUT["coords [B, 3]"]:::input
    INPUT --> PE["Positional Encoding<br/>[B, 39]"]:::op
    INPUT --> FW["fine_weights φ(x)<br/>[B, 32]"]:::op
    FW --> MIX["× V (per-anchor latent)<br/>[F=32, K=32]<br/>= F(x) — mixed latent <b>[B, 32]</b>"]:::op
    PE -->|"[B, 39]"| CAT["concat<br/>[B, 71]"]:::op
    MIX -->|"[B, 32]"| CAT
    CAT --> MLP["fine_mlp · 71→128→128→128→27<br/>GPC Eq.7"]:::mlp
    MLP --> OUT["f_fine [B, 27]"]:::output

    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef output fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    classDef op fill:#fff9c4,stroke:#f57f17
    classDef mlp fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

> [!NOTE]
> **Earlier we tried a different formulation** that fed the 32-dim Gaussian weights $\boldsymbol{\varphi}(\mathbf{x})$ directly into the MLP and pulled $\mathbf{V}\,\boldsymbol{\varphi}$ in via a $0.2\times$ linear side path. With $\mathbf{V} \in \mathbb{R}^{F \times D=27}$ this meant the MLP only ever saw positional information; the scene content reached the output through a non-trainable mixer. Switching to the paper-aligned form above gave **+0.23 dB PSNR** at 3-seed mean (matched compute), with seed-to-seed variance also going down (0.46 → 0.33 dB). See [§ Ablation: Why Each Component Stays](#ablation-why-each-component-stays) for the full v7 numbers.

#### Residual Refiner — 27-dim Cross-Channel Correction

A small MLP `[B,30] → [B,27]` that reads the blended output and patches systematic per-channel biases the fine branch can't see.

> [!IMPORTANT]
> **The refiner is not redundant.** A parameter-matched ablation (3 seeds × 3 variants @ ~80,750 params) shows it contributes **+0.55 to +1.07 dB PSNR** vs. spending the same budget on more fine Gaussians or a wider MLP — see [§ Ablation: Why Each Component Stays](#ablation-why-each-component-stays).

### Training Strategy — Three-Stage Curriculum

```mermaid
gantt
    title Coarse-to-Fine Training (3500 epochs total)
    dateFormat X
    axisFormat %s

    section Stage 1 — Coarse Focus
    MBD branch only · λ_c = 0.5  :s1, 0, 500

    section Stage 2 — Joint Training
    All branches · λ_c decays 0.5→0.1   :s2, 500, 2500

    section Stage 3 — Fine Focus
    Fine + MLP + Refiner · λ_c = 0.1   :s3, 3000, 500
```

| Stage | Epochs | What's optimised | $\lambda_{\text{coarse}}$ |
|:-----:|-------:|------------------|:------------------------:|
| 1 | 500  | Coarse Gaussians + MBD tensors only | 0.5 (high) |
| 2 | 2500 | All branches jointly                 | 0.5 → 0.1 (decay) |
| 3 | 500  | Fine Gaussians + Fine MLP + Refiner  | 0.1 (low) |

### Loss

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{final}} + \lambda_{\text{coarse}} \cdot \mathcal{L}_{\text{coarse}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}
$$

- $\mathcal{L}_{\text{final}}$: reconstruction MSE — channel weights $w_c \propto 1/\sigma_c^2$ balance gradients across SH orders.
- $\mathcal{L}_{\text{coarse}}$: intermediate supervision pinning the MBD branch on the low-frequency target.
- $\mathcal{L}_{\text{reg}}$: scale-norm regulariser preventing Gaussian-scale explosion.

### Ablation: Why Each Component Stays

Five rounds (v2 → v7), 3 seeds × 3500 epochs each, real probe data, FP16 post-training quantization, identical training loop. Each round answers one question.

#### v7: Paper-aligned fine branch beats weights-into-MLP

The current fine branch follows GPC Eq.5/7 exactly (see [§ Fine Branch](#fine-branch--mixed-latent-mlp-paper-aligned-gpc-eq57)). To prove it's an actual improvement over the earlier "feed Gaussian weights into MLP, side-path the latent" formulation, both were trained head-to-head with the same 3-stage curriculum.

| Variant | Fine-branch form | PSNR (mean ± std) | SSIM | Params |
|---|---|---:|:---:|---:|
| OLD | $\text{MLP}(\text{PE}\,\Vert\,\boldsymbol{\varphi}) + 0.2\,\boldsymbol{\varphi}\,\mathbf{V}_{[F,D=27]}$ | 40.317 ± 0.461 | 0.9482 | 80,774 |
| **NEW** | $\text{MLP}\!\big(\text{PE}\,\Vert\,\boldsymbol{\varphi}\,\mathbf{V}_{[F,K=32]}\big)$ | **40.547 ± 0.330** | **0.9495** | 80,934 |
| Δ | | **+0.230 dB** | +0.0013 | +160 |

NEW wins on 2 of 3 seeds and tightens the seed-to-seed spread (0.46 → 0.33 dB), at +0.4 KB FP16 cost.

#### v2: Residual refiner is not "wasted parameters"

We re-spent the refiner's budget on more fine Gaussians or a wider fine MLP — it doesn't recover the gain.

| Variant | Description | PSNR | SSIM | Params |
|---|---|---:|:---:|---:|
| **A**: with refiner | F=32, h=128, +residual_refiner | **40.32 ± 0.49** | **0.9482** | 80,774 |
| B: more fine Gaussians | F=54, h=128, no refiner | 39.25 ± 0.61 | 0.9348 | 80,687 |
| C: wider fine MLP | F=32, h=134, no refiner | 39.85 ± 0.07 | 0.9409 | 80,785 |
| D: fine sees coarse | fine_mlp gets coarse_output, no refiner | 39.55 ± 0.45 | 0.9353 | 80,491 |

The refiner is the only component seeing the 27-dim blended output, so it can fix **cross-channel correlated errors** (e.g. L0_R and L0_G drifting together) that the per-coordinate fine MLP cannot. **+1.07 dB / +0.47 dB / +0.77 dB** over the three same-budget alternatives.

#### v5–v6: PE is structurally essential

| Probe | What was changed (vs production) | PSNR | Δ |
|---|---|---:|---:|
| Production (current) | — | 40.55 | — |
| **G** (v5): drop side path, drop refiner, keep PE | Tests if PE alone covers high-freq | 39.52 | −1.03 |
| **H** (v5): also drop PE | Just MBD + mixed latent + fine MLP | 33.71 | −6.84 |
| I (v6): no PE, F=128 anchors | 4× anchor density to compensate for missing PE | 33.64 | −6.91 |
| I (v6): no PE, F=512 anchors | 16× anchor density | 33.84 | −6.71 |

**PE is irreplaceable** — even 16× anchor density can't recover the 6.7 dB drop. Refiner removal alone costs ~1 dB even with PE. Only the $0.2\times$ side path was safely removable, which is what v7 tests systematically.

<details>
<summary>What each round added (one-liner each)</summary>

- **v2** — disproves "the refiner is redundant" by showing the same parameter budget elsewhere is strictly worse.
- **v4** — first run of paper-aligned fine branch (E_paper_K32) inside the simplified loop; +0.23 dB vs OLD.
- **v5** — ablates the side path and the refiner together; refiner alone removal costs 0.8 dB.
- **v6** — kills the "PE is just dressing, more anchors will replace it" hypothesis (it won't, by 6+ dB).
- **v7** — full buildtime training loop A/B; the +0.23 dB is real, not an artifact of the ablation harness.

</details>

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
.
├── docs/
│   └── architecture.png          # Architecture diagram (project doc)
├── HI-NE-GBD/                    # Production pipeline (use this)
│   ├── buildtime/
│   │   ├── HI-NE-GBD.py          #   Main training script (offline compression)
│   │   └── probe_reader.py       #   Probe data reader utility
│   ├── runtime/
│   │   ├── decoder.py            #   Real-time decoder (Python)
│   │   ├── export_model.py       #   PyTorch -> UCommon .uasset exporter
│   │   └── cpp/                  #   C++ runtime (MSVC)
│   │       ├── HINEGBD.h/cpp     #     decoder library
│   │       ├── main.cpp          #     demo entry point
│   │       └── CMakeLists.txt
│   └── probedata/
│       └── ILCSampleData_0.bin   #   Light probe raw data (only copy)
├── experiments/
│   ├── ablation/                 # Component ablations (mostly synthetic signals)
│   │   ├── HI-NE-GBD.py          #   Real-probe re-train without saving (visualize-only)
│   │   ├── MBD_gaussian.py       #   No-MLP variant (synthetic)
│   │   ├── MBD_gaussian_MLP.py   #   Non-hierarchical variant (synthetic)
│   │   └── gaussian_MLP.py       #   No-MBD variant (synthetic)
│   └── legacy/                   # Earlier exploratory scripts (synthetic only)
└── output/                       # Saved figures from prior runs
```

`compressed_model.pth` (training output) and `hinegbd_model.uasset` (exporter output) are not checked in — they are produced by running the buildtime / export scripts.

## Usage

Scripts use paths relative to `__file__`, so any working directory works.

### 1. Training (Offline Compression)

```bash
python HI-NE-GBD/buildtime/HI-NE-GBD.py
```

Trains for 3500 epochs (500 coarse + 2500 joint + 500 fine) on `HI-NE-GBD/probedata/ILCSampleData_0.bin` and writes `HI-NE-GBD/runtime/compressed_model.pth` (FP16-quantized).

### 2. Real-time Decompression (Python)

```python
from decoder import HINEGBDDecoder

decoder = HINEGBDDecoder("compressed_model.pth")

# Single point query (world coordinates)
sh_coeffs = decoder.decode(x=100.0, y=50.0, z=-200.0)

# Batch query
import numpy as np
coords = np.array([[100, 50, -200], [110, 60, -180]], dtype=np.float32)
sh_batch = decoder.decode_batch(coords)
```

### 3. Interactive / Benchmark / Single-point CLI

```bash
python HI-NE-GBD/runtime/decoder.py --interactive
python HI-NE-GBD/runtime/decoder.py --benchmark
python HI-NE-GBD/runtime/decoder.py --coords 100 50 -200            # world coords
python HI-NE-GBD/runtime/decoder.py --coords 0.5 0.5 0.5 --normalized
```

`--device {cpu,cuda}` overrides auto-detection; `--model PATH` overrides the default `compressed_model.pth`.

### 4. Export to C++ Runtime

```bash
python HI-NE-GBD/runtime/export_model.py \
  --model compressed_model.pth --output hinegbd_model.uasset --verify
```

Then build the C++ demo (Windows / MSVC):

```bash
cmake -S HI-NE-GBD/runtime/cpp -B HI-NE-GBD/runtime/cpp/build
cmake --build HI-NE-GBD/runtime/cpp/build --config Release
HI-NE-GBD/runtime/cpp/build/bin/HINEGBD_Demo.exe hinegbd_model.uasset
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

The `experiments/ablation/` directory contains ablation scripts for individual components (run on synthetic signals from `test_signal_3d.py`, except where noted):
- `MBD_gaussian.py`: MBD + Gaussian (no MLP)
- `MBD_gaussian_MLP.py`: MBD + Gaussian + MLP
- `gaussian_MLP.py`: Gaussian + MLP (no MBD)
- `HI-NE-GBD.py`: hierarchical model run on real probe data (`ILCSampleData_0.bin`)

`experiments/legacy/` keeps earlier exploratory scripts (`opt.py`, `HI_TEST.py`, `MBD_*.py`, `gaussian_control.py`) for historical reference.

