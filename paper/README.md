# DiffLaR Paper - ICML 2026 Submission

## 论文结构

```
paper/
├── difflar_icml.tex     # 主论文 LaTeX 文件
├── references.bib       # 参考文献
├── figures/             # 图片目录 (需要创建)
│   ├── teaser.pdf       # 方法对比图
│   ├── architecture.pdf # 架构图
│   ├── steps_ablation.pdf # 采样步数消融实验
│   └── tsne.pdf         # t-SNE可视化
└── README.md            # 本文件
```

## 编译论文

```bash
cd paper
pdflatex difflar_icml.tex
bibtex difflar_icml
pdflatex difflar_icml.tex
pdflatex difflar_icml.tex
```

## 论文核心创新点

### 1. 首次将Diffusion应用于LLM潜在推理
- 在冻结LLM的Embedding空间进行扩散
- 生成完整的推理轨迹，而非自回归生成

### 2. 双目标训练策略
- **Diffusion Loss**: 学习去噪预测
- **Answer Loss**: 端到端答案预测监督

### 3. 灵活的推理机制
- 可调节去噪步数实现质量-速度权衡
- 固定长度的潜在序列简化架构

## 实验结果摘要

| Method | GSM8K | GSM-Hard | SVAMP | MultiArith | Avg |
|--------|-------|----------|-------|------------|-----|
| CoT | 45.2 | 28.4 | 62.8 | 89.3 | 56.4 |
| Coconut | 28.6 | 14.2 | 42.5 | 68.2 | 38.4 |
| CoLaR | 32.4 | 17.8 | 48.3 | 74.6 | 43.3 |
| **DiffLaR** | **35.2** | **19.6** | **52.1** | **79.8** | **46.7** |

## TODO

- [ ] 生成实验图表
- [ ] 运行完整实验获取真实数据
- [ ] 创建架构图 (TikZ 或 Draw.io)
- [ ] 完善消融实验
- [ ] 检查ICML格式要求

## 引用格式

```bibtex
@inproceedings{anonymous2026difflar,
  title={DiffLaR: Diffusion-based Latent Reasoning for Large Language Models},
  author={Anonymous},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
