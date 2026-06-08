# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research/prototype code for **HI-NE-GBD (Hierarchical Neural Gaussian Basis Decomposition)** — compresses pre-computed light-probe SH coefficients via a Coarse-to-Fine network and decompresses them at runtime in Python or C++. References: *Moving Basis Decomposition for Precomputed Light Transport* (EGSR 2021) and *Gaussian Compression for Precomputed Indirect Illumination* (SIGGRAPH 2025). See `README.md` for architecture, training schedule, and metrics.

There is no `requirements.txt`/`pyproject.toml`, no lint config, no test suite. Dependencies (PyTorch ≥ 1.10, NumPy, SciPy, Matplotlib) install manually. Comments and console output are bilingual Chinese/English.

## Layout

```
.
├── docs/architecture.png   canonical diagram (referenced by README)
├── HI-NE-GBD/              ← production pipeline (the deliverable)
│   ├── buildtime/          training script + probe reader utility
│   ├── runtime/            decoder.py, export_model.py, cpp/
│   └── probedata/          ILCSampleData_0.bin (the only copy)
├── experiments/
│   ├── ablation/           visualization variants (see notes below)
│   └── legacy/             earlier exploratory scripts (synthetic-only)
└── output/                 saved figures from prior runs
```

### What runs on what data

This is the single most useful thing to know before touching a script — the directory tree alone hides it.

| Script | Reads | Saves model? |
|---|---|---|
| `HI-NE-GBD/buildtime/HI-NE-GBD.py` | `HI-NE-GBD/probedata/ILCSampleData_0.bin` (real probes) | **Yes** → `runtime/compressed_model.pth` |
| `experiments/ablation/HI-NE-GBD.py` | same real probe binary (path patched to `../../HI-NE-GBD/probedata/`) | No — only prints metrics + plots |
| `experiments/ablation/MBD_gaussian.py` | synthetic signal `"neon"` from `test_signal_3d.py` | No |
| `experiments/ablation/MBD_gaussian_MLP.py` | synthetic signal | No |
| `experiments/ablation/gaussian_MLP.py` | synthetic signal | No |
| `experiments/legacy/*.py` | synthetic signals only | No |

`experiments/ablation/HI-NE-GBD.py` is **not really an ablation** — it's a near-copy of the buildtime trainer (same hyperparameters: L=16, M=N=64, F=32, MLP hidden=128, PE freqs=6) minus the final model-save block. Treat it as a metrics/visualization scratchpad for the same model. The real ablations are the three sibling scripts (`MBD_gaussian*`, `gaussian_MLP`) on synthetic signals — each removes one component (no MLP / no MBD / non-hierarchical).

`experiments/legacy/` (`opt.py`, `HI_TEST.py`, `MBD_Demo.py`, `MBD_Control.py`, `gaussian_control.py`, `test_signal_3d.py`) is the pre-`HI-NE-GBD/` exploration — historical reference, don't extend.

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
  - Coarse: MBD with M coefficient Gaussians × N basis Gaussians × L bases (real-probe defaults: L=16, M=N=64, D=27). Each Gaussian carries position + log_scale + quaternion + alpha.
  - Fine: F=32 sparse "fine Gaussian" anchors feed weights into an MLP alongside positional encoding (`pe_num_freqs=6`).
  - Residual: small MLP that corrects the blended output.
- **Training** is a hard-coded 3-stage curriculum: 500 / 2500 / 500 epochs (`epochs_coarse=500, epochs_main=2500, epochs_fine=500` at `buildtime/HI-NE-GBD.py:838`). `λ_coarse` decays across stage 2. After training, weights are FP16-quantized.
- **Cross-package import**: `buildtime/HI-NE-GBD.py` saves the model by `sys.path.insert`-ing `runtime/` and calling `decoder.save_compressed_model` — the runtime package is both the consumer and provider of the serialization format.
- **C++ runtime** (`HI-NE-GBD/runtime/cpp/`) consumes the `.uasset` produced by `export_model.py` (UCommon `FFileArchive` format) and depends on `third_party/lib/UCommon_Runtime.lib`. CMake config is **MSVC-only** (`/utf-8`, `/wd4251`, VS output dirs). The `build/` tree is gitignored — generate it locally.

When changing the model layout, four things must move together: training (`HI-NE-GBD/buildtime/HI-NE-GBD.py`), Python decoder (`HI-NE-GBD/runtime/decoder.py`), exporter (`HI-NE-GBD/runtime/export_model.py`), and C++ struct layout (`HI-NE-GBD/runtime/cpp/HINEGBD.{h,cpp}`). The exporter writes a positional tensor stream that `HINEGBD::FHINEGBDModel::Load` reads in order.

## Generated artifacts (not in git)

`.gitignore` excludes the regenerable products:
- `*.pth` — `HI-NE-GBD/runtime/compressed_model.pth` (output of buildtime)
- `*.uasset` — `HI-NE-GBD/runtime/cpp/hinegbd_model.uasset` (output of export_model.py)
- `HI-NE-GBD/runtime/cpp/build/` — CMake/MSVC tree
- `__pycache__/`, `*.log`, `opt_output.txt`

Fresh clone needs to run buildtime + export before C++ demo or `decoder.py --interactive` will work.

## Commands

All Python scripts use hardcoded paths relative to `__file__`, so they can be invoked from any CWD.

```bash
# Train (writes ../runtime/compressed_model.pth, takes 3500 epochs)
python HI-NE-GBD/buildtime/HI-NE-GBD.py

# Real-time decode in Python (needs compressed_model.pth)
python HI-NE-GBD/runtime/decoder.py --interactive
python HI-NE-GBD/runtime/decoder.py --benchmark
python HI-NE-GBD/runtime/decoder.py --coords 100 50 -200            # world coords
python HI-NE-GBD/runtime/decoder.py --coords 0.5 0.5 0.5 --normalized
# Flags: --model PATH, --device {cpu,cuda} (auto-detected default)

# Export PyTorch → UCommon .uasset for C++ runtime (run from runtime/ for default paths)
python HI-NE-GBD/runtime/export_model.py --model compressed_model.pth --output hinegbd_model.uasset --verify

# Re-train+visualize without saving (real probes, same hyperparams as production)
python experiments/ablation/HI-NE-GBD.py

# Component ablations (synthetic signals — no probe data needed)
python experiments/ablation/MBD_gaussian.py        # MBD only, no MLP
python experiments/ablation/gaussian_MLP.py        # No MBD, just Gaussian + MLP
python experiments/ablation/MBD_gaussian_MLP.py    # All components, non-hierarchical

# C++ runtime (Windows / MSVC only — build tree is gitignored)
cmake -S HI-NE-GBD/runtime/cpp -B HI-NE-GBD/runtime/cpp/build
cmake --build HI-NE-GBD/runtime/cpp/build --config Release
HI-NE-GBD/runtime/cpp/build/bin/HINEGBD_Demo.exe hinegbd_model.uasset
HI-NE-GBD/runtime/cpp/build/bin/HINEGBD_Demo.exe hinegbd_model.uasset 100 50 -200
```

There are no tests to run, no linter, and no formatter configured.
