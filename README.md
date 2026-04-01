# Control-Note

[中文 README](./README_zh.md)

Control-Note is a practical control-theory note repository for robotics competition students.
It includes English and Chinese LaTeX notes, generated figures, and helper scripts for building illustrations.

## Repository Overview

- Practical control theory notes focused on engineering use cases.
- Two language versions of the document:
  - English: `main.tex`
  - Chinese: `main_zh.tex` + chapter files in `sections_zh/`
- Generated PDF outputs in `build/`.
- Python figure generation scripts in `scripts/`.

## File Structure

```text
Control-Note/
├─ main.tex                 # English full note (single-file LaTeX source)
├─ main_zh.tex              # Chinese main entry LaTeX source
├─ sections_zh/             # Chinese chapter split files
├─ include/
│  └─ control/              # Supporting include resources
├─ figures/                 # English figures
├─ figures_zh/              # Chinese figures
├─ scripts/                 # Figure generation scripts (EN + ZH variants)
├─ build/
│  ├─ Control_Theory_Note.pdf
│  └─ Control_Theory_Note_cn.pdf
└─ README_zh.md             # Chinese README
```

## Notes

- This repository is documentation-oriented (LaTeX + figures).
- If you want to read Chinese content, jump to [中文 README](./README_zh.md).
