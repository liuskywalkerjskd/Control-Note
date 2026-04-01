# A Practical Guide to Control Theory

[中文 README](./README_zh.md)

**Author:** Skywalker Liu  
**Language:** English + Chinese (Simplified)  
**Format:** LaTeX (XeLaTeX)  
**License:** Educational use

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
| 1 | **Introduction** | Motivation, learning path, how to use this guide |
| 2 | **Digital Signal Processing** | Frequency-domain basics, Laplace/Fourier transforms, LPF/complementary/notch filters, sensor noise |
| 3 | **System Description** | Transfer functions, state-space models, poles & zeros, stability, time/frequency-domain analysis |
| 4 | **Classical Control** | PID (single/cascade/parallel), tuning methods, feedforward, trajectory generation |
| 5 | **Discretization & Implementation** | ZOH, Tustin, matched Z-transform, embedded implementation concerns |
| 6 | **Modern Control** | State feedback, LQR, observability, Kalman filter, EKF, LQG, linear MPC |
| 7 | **Future Prospects** | Data-driven control, DeePC, reinforcement learning, iLQR/DDP, learning + control |
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

### Local Build

Install TeX Live with CJK support, then run:

```bash
xelatex main_zh.tex
xelatex main_zh.tex
```

You can similarly build the English document with:

```bash
xelatex main.tex
xelatex main.tex
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
