# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research/prototype code for **HI-NE-GBD (Hierarchical Neural Gaussian Basis Decomposition)** — compresses pre-computed light-probe SH coefficients via a Coarse-to-Fine network and decompresses them at runtime in Python or C++. References: *Moving Basis Decomposition for Precomputed Light Transport* (EGSR 2021) and *Gaussian Compression for Precomputed Indirect Illumination* (SIGGRAPH 2025). See `README.md` for architecture, training schedule, and metrics.

There is no `requirements.txt`/`pyproject.toml`, no lint config, no test suite. Dependencies (PyTorch ≥ 1.10, NumPy, SciPy, Matplotlib) install manually. Comments and console output are bilingual Chinese/English.

## Layout — two trees, only one ships

```
.
├── docs/architecture.png  canonical architecture diagram (referenced by README)
├── HI-NE-GBD/             ← production pipeline (the deliverable)
│   ├── buildtime/         train + probe reader
│   ├── runtime/           decoder.py, export_model.py, cpp/
│   └── probedata/         ILCSampleData_0.bin (only copy in repo)
├── experiments/
│   ├── ablation/          component ablation studies + per-experiment .png
│   └── legacy/            earlier exploratory scripts (kept as reference)
└── output/                saved figures from prior runs
```

- **`HI-NE-GBD/` is where new work belongs.** Don't add ablation/exploration code here.
- **`experiments/ablation/`** mostly runs on synthetic signals from `test_signal_3d.py` (`get_test_signal_by_name("sunset", grid_size=32, num_channels=3)`). The exception is `experiments/ablation/HI-NE-GBD.py`, which reads the real probe binary — its parameters (e.g. `num_bases=8, coeff_res=128`) are explicitly comment-tagged "与 HI-NE-GBD 完整模型对齐", so it tracks the production model.
- **`experiments/legacy/`** is the pre-`HI-NE-GBD/` exploration (`opt.py`, `HI_TEST.py`, `MBD_Demo.py`, `MBD_Control.py`, `gaussian_control.py`, `test_signal_3d.py`). These run on synthetic signals only. Treat as historical unless explicitly asked.

`output/` holds figures from prior runs (Control_group_PSNR/, Real-Probe-Data/, ablation/, …).

## The HI-NE-GBD pipeline

```
ILCSampleData_0.bin  ──►  HI-NE-GBD/buildtime/HI-NE-GBD.py  ──►  HI-NE-GBD/runtime/compressed_model.pth (regenerable, gitignored)
                                                                    │
                                          ┌─────────────────────────┼──────────────────────────┐
                                          ▼                                                    ▼
                          HI-NE-GBD/runtime/decoder.py        HI-NE-GBD/runtime/export_model.py
                          (Python real-time decode)                       │
                                                                          ▼
                                                         hinegbd_model.uasset (regenerable, gitignored)
                                                                          │
                                                                          ▼
                                              HI-NE-GBD/runtime/cpp/HINEGBD.{h,cpp} + main.cpp
                                              (C++ runtime, links UCommon_Runtime.lib)
```

- **Probe binary format** is fixed: 32 floats per probe — `position[3], radius, sh_coeffs[27] (RGB-interleaved across 9 SH bands), shadow`. `load_ilc_probe_data()` (`HI-NE-GBD/buildtime/HI-NE-GBD.py:39`) is the canonical reader; positions are normalized to `[0, 1]` per axis and `pos_min`/`pos_max` are stored in the model so decoders can de-normalize world coordinates.
- **Model architecture** (`f_final(x) = f_coarse(x) + f_fine(x) + 0.1 * residual(x)`):
  - Coarse: MBD with M coefficient Gaussians × N basis Gaussians × L bases (defaults: L=16, M=N=64, D=27). Each Gaussian carries position+log_scale+quaternion+alpha.
  - Fine: F=32 sparse "fine Gaussian" anchors feed weights into an MLP alongside positional encoding (`pe_num_freqs=6`).
  - Residual: small MLP that corrects the blended output.
- **Training** is a hard-coded 3-stage curriculum (500 / 2500 / 500 epochs) inside `HI-NE-GBD.py`; `λ_coarse` decays across stage 2. Output is FP16-quantized.
- **C++ runtime** (`HI-NE-GBD/runtime/cpp/`) consumes a `.uasset` produced by `export_model.py` (UCommon `FFileArchive` format) and depends on `third_party/lib/UCommon_Runtime.lib`. The CMake config is **MSVC-only** (`/utf-8`, `/wd4251`, VS output dirs). The `build/` directory is gitignored — generate it locally with CMake.

When changing the model layout, four things must move together: training (`HI-NE-GBD/buildtime/HI-NE-GBD.py`), Python decoder (`HI-NE-GBD/runtime/decoder.py`), exporter (`HI-NE-GBD/runtime/export_model.py`), and C++ struct layout (`HI-NE-GBD/runtime/cpp/HINEGBD.{h,cpp}`). The exporter writes a positional tensor stream that `HINEGBD::FHINEGBDModel::Load` reads in order — order matters.

## Generated artifacts (not in git)

`.gitignore` excludes the regenerable products:
- `*.pth` — training output (`HI-NE-GBD/runtime/compressed_model.pth`)
- `*.uasset` — exporter output (`HI-NE-GBD/runtime/cpp/hinegbd_model.uasset`)
- `HI-NE-GBD/runtime/cpp/build/` — CMake/MSVC build tree
- `__pycache__/`, `*.log`, `opt_output.txt`

If a fresh clone hits "model file not found", run buildtime + export first.

## Commands

All Python scripts use hardcoded paths relative to `__file__`, so they can be invoked from any working directory.

```bash
# Train (writes ../runtime/compressed_model.pth)
python HI-NE-GBD/buildtime/HI-NE-GBD.py

# Real-time decode in Python
python HI-NE-GBD/runtime/decoder.py --interactive
python HI-NE-GBD/runtime/decoder.py --benchmark
python HI-NE-GBD/runtime/decoder.py --coords 100 50 -200            # world coords
python HI-NE-GBD/runtime/decoder.py --coords 0.5 0.5 0.5 --normalized
# Flags: --model PATH, --device {cpu,cuda} (auto-detected)

# Export PyTorch → UCommon .uasset for the C++ runtime
python HI-NE-GBD/runtime/export_model.py --model compressed_model.pth --output hinegbd_model.uasset --verify

# Ablation experiments (synthetic signals; HI-NE-GBD.py is the one that uses real probe data)
python experiments/ablation/HI-NE-GBD.py
python experiments/ablation/MBD_gaussian.py
python experiments/ablation/gaussian_MLP.py
python experiments/ablation/MBD_gaussian_MLP.py

# C++ runtime (Windows / MSVC only — build tree is not in git)
cmake -S HI-NE-GBD/runtime/cpp -B HI-NE-GBD/runtime/cpp/build
cmake --build HI-NE-GBD/runtime/cpp/build --config Release
HI-NE-GBD/runtime/cpp/build/bin/HINEGBD_Demo.exe hinegbd_model.uasset
HI-NE-GBD/runtime/cpp/build/bin/HINEGBD_Demo.exe hinegbd_model.uasset 100 50 -200
```

There are no tests to run, no linter, and no formatter configured.
