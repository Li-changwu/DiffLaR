# Direct Latent Diffusion with Self-Conditioning for LLM Reasoning

**Target Venues**: NeurIPS 2025 / ICML 2025

---

## Abstract (250 words)

Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities, yet their autoregressive nature inherently limits parallel computation and iterative refinement of intermediate reasoning steps. Recent work on latent diffusion for reasoning (LaDiR) addresses this by encoding thought tokens into a compressed latent space via a Variational Autoencoder (VAE), then applying flow matching to generate reasoning latents. However, this approach introduces additional architectural complexity and potential information loss during VAE compression.

In this paper, we propose **DiffLaR-SC** (Diffusion Latent Reasoning with Self-Conditioning), a simplified yet effective framework that performs diffusion directly in the LLM's embedding space without requiring a separate VAE. Our key contributions are threefold: (1) We demonstrate that direct embedding-space diffusion is not only feasible but achieves comparable or superior performance to VAE-based approaches, significantly reducing architectural complexity. (2) We introduce a novel fusion of Self-Conditioning with two-stage Rollout training, where Self-Conditioning provides step-wise consistency during denoising while Rollout training bridges the train-inference distribution gap. (3) We provide theoretical analysis showing that Self-Conditioning acts as an implicit regularizer, improving convergence stability under the flow matching objective.

Extensive experiments on mathematical reasoning benchmarks (GSM8K, MATH) demonstrate that DiffLaR-SC achieves **XX.X%** accuracy, outperforming both standard Chain-of-Thought fine-tuning and prior latent diffusion methods. Ablation studies reveal that the Self-Conditioning mechanism contributes **X.X%** improvement, while the simplified architecture reduces training time by **XX%** compared to VAE-based alternatives. Our code and models are publicly available.

---

## 1. Introduction

### 1.1 开篇：问题引入 (1段)

Large Language Models (LLMs) have achieved remarkable success across various reasoning tasks through Chain-of-Thought (CoT) prompting and fine-tuning [Wei et al., 2022]. However, the autoregressive generation paradigm presents fundamental limitations: (1) sequential token generation prevents parallel computation of reasoning steps, (2) errors in early tokens propagate irreversibly to subsequent generations, and (3) the model cannot iteratively refine its intermediate reasoning.

### 1.2 现有工作及局限 (2段)

Recent advances in latent reasoning aim to address these limitations by operating in continuous latent spaces rather than discrete token spaces. Coconut [Hao et al., 2024] pioneered this direction by replacing explicit CoT tokens with learnable latent embeddings. More recently, LaDiR [Liu et al., 2025] introduced a latent diffusion framework that encodes thought tokens via a VAE and generates reasoning latents through flow matching, achieving state-of-the-art results on mathematical reasoning benchmarks.

Despite these advances, current latent diffusion approaches face several challenges:
- **Architectural Complexity**: VAE-based methods require training a separate encoder-decoder, introducing additional hyperparameters and potential training instabilities.
- **Information Bottleneck**: Compressing variable-length thought sequences into fixed-dimensional latents may lose fine-grained reasoning information.
- **Distribution Shift**: A critical gap exists between training (using ground-truth latents) and inference (using self-generated latents), leading to error accumulation.

### 1.3 我们的方案 (1段)

In this work, we propose DiffLaR-SC, a simplified framework that addresses these challenges through two key innovations:

1. **Direct Embedding-Space Diffusion**: We demonstrate that diffusion can be performed directly in the LLM's embedding space without VAE compression. By carefully designing the normalization scheme and flow matching objective, we achieve stable training while preserving the full expressiveness of LLM embeddings.

2. **Self-Conditioning with Rollout Fusion**: We introduce a novel training strategy that combines Self-Conditioning—using the model's own predictions as additional input—with two-stage Rollout training. In Stage 1, Self-Conditioning is applied probabilistically (50%) to learn robust velocity predictions. In Stage 2, we transition to Rollout training where the model learns from its own generated latents, with Self-Conditioning applied deterministically (100%) to maximize prediction consistency.

### 1.4 贡献总结 (1段)

Our main contributions are:

- **Simplified Architecture**: We show that VAE-free latent diffusion is not only feasible but often preferable, reducing model complexity while maintaining or improving performance.

- **Novel Training Strategy**: We propose the first fusion of Self-Conditioning with Rollout training for latent reasoning, providing both theoretical justification and empirical validation.

- **Theoretical Insights**: We analyze Self-Conditioning under the flow matching framework, proving that it acts as an implicit variance reduction mechanism that stabilizes training.

- **State-of-the-Art Results**: Experiments on GSM8K and MATH demonstrate XX.X% and XX.X% accuracy respectively, with XX% faster training compared to VAE-based alternatives.

---

## 2. Related Work

### 2.1 Chain-of-Thought Reasoning
- Wei et al. (2022): CoT prompting
- Kojima et al. (2022): Zero-shot CoT
- Wang et al. (2023): Self-consistency

### 2.2 Latent Reasoning
- Deng et al. (2024): Implicit CoT
- Hao et al. (2024): Coconut - continuous latent reasoning
- Liu et al. (2025): LaDiR - latent diffusion for reasoning

### 2.3 Diffusion Models
- Ho et al. (2020): DDPM
- Song et al. (2021): Score-based models
- Lipman et al. (2023): Flow matching
- Chen et al. (2024): Self-conditioning in diffusion

---

## 3. Preliminaries

### 3.1 Problem Formulation

Given a question $q$ and ground-truth reasoning steps $s = (s_1, ..., s_n)$ leading to answer $a$, we aim to learn a model that generates reasoning latents $z \in \mathbb{R}^{L \times d}$ such that conditioning on $z$ enables the LLM to produce the correct answer.

### 3.2 Flow Matching

Flow matching defines a probability path from data distribution $p_0$ to noise distribution $p_1$ via linear interpolation:

$$z_t = (1-t) \cdot z_0 + t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The velocity field is defined as:

$$u_t^*(z_t | z_0) = \epsilon - z_0$$

The model $v_\theta(z_t, t)$ is trained to predict this velocity:

$$\mathcal{L}_{FM} = \mathbb{E}_{t, z_0, \epsilon} \left[ \| v_\theta(z_t, t) - u_t^* \|^2 \right]$$

### 3.3 Self-Conditioning

Self-Conditioning [Chen et al., 2024] augments the model input with its own prediction from the previous step:

$$v_\theta(z_t, t, \hat{z}_0^{(prev)})$$

where $\hat{z}_0^{(prev)}$ is estimated from the previous velocity prediction.

---

## 4. Method: DiffLaR-SC

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      DiffLaR-SC Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Question ──→ [LLM Embed] ──→ Query Embedding (condition)   │
│                                      ↓                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Latent Diffusion Module                │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │ z_T ~ N(0,I)                                │   │   │
│  │  │      ↓                                      │   │   │
│  │  │ for t = 1.0 → 0.0:                         │   │   │
│  │  │   v = Denoiser(z_t, t, query, z0_cond)     │   │   │ ← Self-Cond
│  │  │   z_t = z_t - dt * v        (Euler step)   │   │   │
│  │  │      ↓                                      │   │   │
│  │  │ z_0 = denormalize(z_t)                     │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  Steps Embedding ──→ [Concat] ──→ LLM.generate() ──→ Answer │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Direct Embedding-Space Diffusion

Unlike VAE-based approaches, we perform diffusion directly on LLM embeddings. The key challenge is that LLM embeddings are not naturally normalized, which can cause training instabilities.

**Normalization Strategy**:
Given training embeddings $\{e_i\}_{i=1}^N$, we estimate:
- Per-dimension mean: $\mu = \frac{1}{N} \sum_i e_i$
- Per-dimension std: $\sigma = \sqrt{\frac{1}{N} \sum_i (e_i - \mu)^2}$
- Global scale: $s = \frac{1}{\max(\sigma)}$

Normalized embedding: $z = s \cdot \frac{e - \mu}{\sigma}$

This ensures $z \approx \mathcal{N}(0, 1)$, compatible with the flow matching formulation.

### 4.3 Denoiser with Self-Conditioning

The denoiser network is a Transformer that takes:
- Noisy latent $z_t \in \mathbb{R}^{L \times d}$
- Time embedding $\gamma(t) \in \mathbb{R}^d$
- Query condition $c \in \mathbb{R}^{L_q \times d}$
- Self-conditioning input $\hat{z}_0 \in \mathbb{R}^{L \times d}$ (optional)

**Self-Conditioning Integration**:
$$h = \text{InputProj}(z_t) + \text{SC-Proj}(\hat{z}_0)$$

where SC-Proj is a learnable projection, and $\hat{z}_0 = \mathbf{0}$ when Self-Conditioning is not applied.

### 4.4 Two-Stage Training with Self-Conditioning Fusion

**Stage 1: Teacher-Forcing with Probabilistic Self-Conditioning**

In Stage 1, we train with ground-truth latents but introduce Self-Conditioning with probability $p_{sc}^{(1)} = 0.5$:

$$\mathcal{L}_1 = \mathbb{E}_{t, z_0^{GT}} \left[ \| v_\theta(z_t, t, c, \hat{z}_0) - ({\epsilon} - z_0^{GT}) \|^2 \right]$$

where $\hat{z}_0$ is computed via:
- With probability 0.5: $\hat{z}_0 = z_t - t \cdot v_\theta(z_t, t, c, \mathbf{0})$ (detached)
- With probability 0.5: $\hat{z}_0 = \mathbf{0}$

**Stage 2: Rollout Training with Deterministic Self-Conditioning**

In Stage 2, we progressively replace ground-truth latents with self-generated ones:

$$\mathcal{L}_2 = r(e) \cdot \mathcal{L}_{rollout} + (1 - r(e)) \cdot \mathcal{L}_{GT}$$

where:
- $r(e) = r_{start} + (r_{end} - r_{start}) \cdot \frac{e - E_1}{E_{total} - E_1}$ is the rollout ratio (curriculum)
- $\mathcal{L}_{rollout}$ uses self-generated $\tilde{z}_0 = \text{Generate}(c)$
- Self-Conditioning is applied with probability $p_{sc}^{(2)} = 1.0$

**Answer Loss**:
Both stages include an answer prediction loss:
$$\mathcal{L}_{ans} = -\log p_{LLM}(a | q, \tilde{z}_0)$$

**Total Loss**:
$$\mathcal{L} = \lambda_{diff} \cdot \mathcal{L}_{diff} + \lambda_{ans} \cdot \mathcal{L}_{ans}$$

### 4.5 Inference

At inference time, we perform Euler integration with Self-Conditioning:

```python
z_T ~ N(0, I)
z0_pred = None

for t in [1.0, 0.9, ..., 0.0]:
    v = Denoiser(z_t, t, query, z0_pred)
    z0_pred = z_t - t * v  # Update Self-Cond
    z_t = z_t - dt * v     # Euler step

return denormalize(z_t)
```

---

## 5. Theoretical Analysis

### 5.1 Self-Conditioning as Variance Reduction

**Theorem 1** (Informal): Under mild regularity conditions, Self-Conditioning reduces the variance of the velocity prediction by a factor of $(1 - \rho^2)$, where $\rho$ is the correlation between consecutive predictions.

**Proof Sketch**: [详细证明]

### 5.2 Convergence Guarantee

**Theorem 2**: The two-stage training with curriculum rollout converges to a stationary point of the population loss under standard assumptions.

---

## 6. Experiments

### 6.1 Setup
- **Datasets**: GSM8K, MATH, AQuA
- **Backbone**: Llama-3.2-1B-Instruct
- **Baselines**: CoT Fine-tuning, Coconut, LaDiR

### 6.2 Main Results
[表格：各方法在各数据集上的准确率]

### 6.3 Ablation Studies
- Self-Conditioning概率的影响
- Rollout比例Curriculum的影响
- 直接嵌入 vs VAE嵌入对比
- 推理步数的影响

### 6.4 Analysis
- 训练稳定性对比（Loss曲线）
- 生成latent的质量分析（Norm分布）
- 推理速度对比

---

## 7. Conclusion

We presented DiffLaR-SC, a simplified latent diffusion framework for LLM reasoning that eliminates the need for VAE compression while introducing a novel fusion of Self-Conditioning with Rollout training. Our approach achieves state-of-the-art performance on mathematical reasoning benchmarks with reduced architectural complexity and training time. We hope this work inspires further research into efficient latent reasoning methods.

---

## Appendix

### A. Implementation Details
### B. Additional Experiments
### C. Proof of Theorems
### D. Broader Impact Statement
