# A Practical Guide to Control Theory

[中文 README](./README_zh.md)

---

## What Is This?

This repository is a practical control-theory note aimed at real robotics engineering work.
It is written around one core goal: helping you design, tune, and debug controllers on real hardware.

Rather than treating theory and application separately, the notes continuously answer:
**"What is this useful for when the robot is actually running?"**

## Who Is This For?

Undergraduate students (roughly year 1-3) with basic calculus and linear algebra.
If you can follow derivatives and matrix multiplication, you are ready.

## Table of Contents

| # | Chapter | Key Topics |
|---|---------|------------|
| - | **Two Numbers: $e$ and $\pi$** | Why $\pi$ governs periodicity and $e$ governs decay; Euler's formula unifies both |
| 1 | **Introduction** | Motivation, learning path, how to use this guide |
| 2 | **Digital Signal Processing** | Frequency-domain basics, Laplace/Fourier transforms, poles & zeros intro, LPF/BPF/HPF/notch filters, Z-transform |
| 3 | **System Description** | Transfer functions, state-space models, stability, 2nd-order systems, DC motor modeling (brushed & BLDC/FOC) |
| 4 | **Classical Control** | Feedback fundamentals, Bode plots, PID (single/cascade/parallel), tuning methods, feedforward, system identification |
| 5 | **Discretization & Implementation** | ZOH, Tustin, matched Z-transform, FIR/IIR filter design, fixed-point, embedded implementation |
| 6 | **Modern Control** | Controllability/observability, LQR, MPC, TinyMPC, Luenberger observer, Kalman/EKF, LQG, trajectory planning |
| 7 | **Attitude & Rotation** | Coordinate frames (NWU), rotation matrices, Euler angles, gimbal lock, axis-angle, quaternions |
| 8 | **Outlook** | Data-driven control, Koopman operator, adaptive control, RL, diffusion policies, VLAs, neural ODEs, event-triggered control |
| A | **Appendix: Plug-and-Play C++ Modules** | Header-only, embedded-friendly algorithm modules |

## Engineering-Workflow-Oriented Structure

The notes are organized by the actual development flow:

1. Clean data (DSP)
2. Build system models
3. Design controllers (PID first)
4. Implement on MCU (discretization)
5. Upgrade with modern control
6. Explore frontier methods

## Project Structure

```text
Control-Note/
├─ main.tex                  # English main document
├─ main_zh.tex               # Chinese main document
├─ sections_zh/              # Chinese chapter source files
├─ include/                  # Shared include resources
├─ figures/                  # English figures
├─ figures_zh/               # Chinese figures
├─ scripts/                  # Figure generation scripts
├─ build/                    # Built PDF outputs
├─ README.md                 # English README
└─ README_zh.md              # Chinese README
```

## How to Build

### Overleaf

Import this repository and compile with **XeLaTeX**.

### Local Build (Recommended)

A build script is provided for local compilation. Prerequisites:

1. **Install TeX Live** (user-space, no sudo required):
   ```bash
   # Download the installer
   cd /tmp && wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
   tar xzf install-tl-unx.tar.gz && cd install-tl-*/

   # Run the installer (choose "small" scheme to save space)
   perl install-tl

   # After installation, add TeX Live to your PATH (adjust the year as needed)
   echo 'export PATH="$HOME/texlive/2026/bin/x86_64-linux:$PATH"' >> ~/.bashrc
   source ~/.bashrc

   # Install required packages
   tlmgr install latexmk ctex mdframed zref needspace booktabs enumitem float caption
   ```

2. **Build PDFs** using the provided script:
   ```bash
   # Build both English and Chinese PDFs
   ./scripts/build.sh all

   # Build English only
   ./scripts/build.sh en

   # Build Chinese only
   ./scripts/build.sh zh

   # Clean auxiliary files
   ./scripts/build.sh clean
   ```

   Output files:
   - `build/Control_Theory_Note.pdf` (English)
   - `build/Control_Theory_Note_cn.pdf` (Chinese)

### Manual Build

If you prefer to compile manually:

```bash
# English (pdflatex)
latexmk -pdf -interaction=nonstopmode main.tex

# Chinese (xelatex)
latexmk -xelatex -interaction=nonstopmode main_zh.tex
```

## Regenerate Figures

```bash
pip install matplotlib numpy
python scripts/generate_figures_zh.py
python scripts/generate_advanced_figures_zh.py
```

See `scripts/` for the complete list of figure generation scripts.

## Contributing

Issues and pull requests are welcome for typo fixes, technical corrections, and structural improvements.
