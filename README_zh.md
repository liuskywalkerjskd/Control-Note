# 控制理论实用指南

[English README](./README.md)

**作者：** Skywalker Liu  
**语言：** 英文 + 简体中文  
**格式：** LaTeX (XeLaTeX)  
**许可：** 仅供学习使用

---

## 这是什么？

这是一个面向真实机器人工程实践的控制理论笔记仓库。
核心目标是：帮助你在真实硬件上设计、调参、排查控制器问题。

内容编排始终围绕一个问题展开：  
**“机器人真正上场时，这个理论到底有什么用？”**

## 写给谁看？

主要面向大一到大三本科生，前置知识是基础微积分和线性代数。  
只要你能理解导数和矩阵乘法，就可以开始。

## 目录

| # | 章节 | 主要内容 |
|---|------|----------|
| 1 | **引言** | 写作动机、学习路径、阅读方式 |
| 2 | **数字信号处理** | 频域基础、拉普拉斯/傅里叶变换、低通/互补/陷波滤波、传感器噪声 |
| 3 | **系统描述** | 传递函数、状态空间、极点零点、稳定性、时域/频域分析 |
| 4 | **经典控制理论** | PID（单环/串级/并联）、调参方法、前馈、轨迹生成 |
| 5 | **离散化与实现** | ZOH、Tustin、匹配 Z 变换、嵌入式实现注意事项 |
| 6 | **现代控制理论** | 状态反馈、LQR、可观性、卡尔曼滤波、EKF、LQG、线性 MPC |
| 7 | **未来展望** | 数据驱动控制、DeePC、强化学习、iLQR/DDP、学习与控制融合 |
| 附 | **附录：即插即用 C++ 模块** | 仅头文件、嵌入式友好的算法模块 |

## 按工程工作流组织

内容顺序贴近实际开发流程：

1. 先清理数据（DSP）
2. 再建立系统模型
3. 然后设计控制器（先从 PID 开始）
4. 再部署到 MCU（离散化）
5. 再用现代控制提升性能
6. 最后了解前沿方向

## 项目结构

```text
Control-Note/
├─ main.tex                  # 英文主文档
├─ main_zh.tex               # 中文主文档
├─ sections_zh/              # 中文分章节源码
├─ include/                  # 共享 include 资源
├─ figures/                  # 英文图表
├─ figures_zh/               # 中文图表
├─ scripts/                  # 图表生成脚本
├─ build/                    # 编译输出 PDF
├─ README.md                 # 英文 README
└─ README_zh.md              # 中文 README
```

## 如何编译

### Overleaf

直接导入仓库，并使用 **XeLaTeX** 编译。

### 本地编译

安装带 CJK 支持的 TeX Live 后执行：

```bash
xelatex main_zh.tex
xelatex main_zh.tex
```

英文版也可使用：

```bash
xelatex main.tex
xelatex main.tex
```

## 重新生成图表

```bash
pip install matplotlib numpy
python scripts/generate_figures_zh.py
python scripts/generate_advanced_figures_zh.py
```

完整脚本列表请查看 `scripts/` 目录。

## 参与贡献

欢迎通过 Issue 或 Pull Request 提交错别字修正、技术勘误和结构优化建议。
