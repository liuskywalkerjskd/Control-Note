# Control-Note

[English README](./README.md)

Control-Note 是一个面向机器人大赛学生的控制理论实用笔记仓库。
仓库包含中英文 LaTeX 文档、配套图像，以及用于生成图像的辅助脚本。

## 仓库简介

- 以工程落地为导向的控制理论笔记。
- 文档提供中英文两个版本：
  - 英文：`main.tex`
  - 中文：`main_zh.tex` + `sections_zh/` 分章节内容
- 生成后的 PDF 位于 `build/`。
- 图像生成脚本位于 `scripts/`。

## 文件架构

```text
Control-Note/
├─ main.tex                 # 英文版完整笔记（单文件 LaTeX 源码）
├─ main_zh.tex              # 中文版主入口 LaTeX 源码
├─ sections_zh/             # 中文版分章节文件
├─ include/
│  └─ control/              # 支撑性 include 资源
├─ figures/                 # 英文版图像
├─ figures_zh/              # 中文版图像
├─ scripts/                 # 图像生成脚本（中英双版本）
├─ build/
│  ├─ Control_Theory_Note.pdf
│  └─ Control_Theory_Note_cn.pdf
└─ README.md                # 英文 README
```

## 说明

- 本仓库以文档内容为主（LaTeX + 图像资源）。
- 如需英文说明，请跳转到 [English README](./README.md)。
