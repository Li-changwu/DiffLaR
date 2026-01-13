# DiffLaR Fused 技术说明书

> **Diffusion-based Latent Reasoning with Fused Training**
> 
> 版本：1.0 | 更新日期：2026-01-09

---

## 目录

1. [算法概述](#1-算法概述)
2. [系统架构](#2-系统架构)
3. [核心组件详解](#3-核心组件详解)
4. [训练阶段](#4-训练阶段)
5. [推理阶段](#5-推理阶段)
6. [数据流详解](#6-数据流详解)
7. [关键技术](#7-关键技术)
8. [配置参数](#8-配置参数)
9. [损失函数](#9-损失函数)

---

## 1. 算法概述

### 1.1 核心思想

DiffLaR Fused 是一种**思维链加速算法**，核心目标是用 Diffusion 模型**并行生成**思考过程的隐表示，替代传统 LLM 的**逐 token 自回归生成**，从而实现思维链推理的加速。

```
传统思维链：
  Question → LLM(token by token) → "Step 1...Step 2...Step 3..." → Answer
  时间复杂度：O(n)，n = 思考步骤的 token 数量

DiffLaR Fused：
  Question → Diffusion(并行采样) → Thinking Embedding → LLM → Answer
  时间复杂度：O(k)，k = 扩散采样步数（固定，通常 10-128 步）
```

### 1.2 关键创新

| 创新点 | 描述 |
|-------|------|
| **Latent Diffusion** | 在 LLM 的 Embedding 空间进行扩散，而非像素或 token 空间 |
| **两阶段训练** | Stage 1 学习基础扩散，Stage 2 引入 Rollout 强化 |
| **Self-Conditioning** | 将上一步预测作为条件，提高生成质量 |
| **中间监督** | Alignment Loss 显式对齐生成 latent 与 GT |
| **Exposure Bias 修复** | 训练时也使用生成 embedding，避免训练-推理分布错配 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DiffLaR Fused 系统架构                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐                                                      │
│   │   Question   │                                                      │
│   │   (文本)      │                                                      │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐                                                      │
│   │  Tokenizer   │                                                      │
│   │  + Embedding │  ◄─── LLM 的 Embedding Layer（冻结）                  │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────┐     Query Embedding [B, L_q, H]                      │
│   │    Query     │────────────────────────────────┐                     │
│   │  Embedding   │                                │                     │
│   └──────────────┘                                │                     │
│                                                   │                     │
│                                                   ▼                     │
│                                    ┌──────────────────────────┐         │
│                                    │   Latent Diffusion Model │         │
│                                    │   ┌──────────────────┐   │         │
│                                    │   │    Denoiser      │   │         │
│                                    │   │  (Transformer)   │   │         │
│                                    │   └──────────────────┘   │         │
│                                    │   ┌──────────────────┐   │         │
│                                    │   │ Noise Scheduler  │   │         │
│                                    │   └──────────────────┘   │         │
│                                    └───────────┬──────────────┘         │
│                                                │                        │
│                                                ▼                        │
│                                    Steps Embedding [B, L, H]            │
│                                    （思考过程的隐表示）                    │
│                                                │                        │
│   ┌──────────────┐                             │                        │
│   │    Query     │                             │                        │
│   │  Embedding   │─────────────────────────────┤                        │
│   └──────────────┘                             │                        │
│                                                ▼                        │
│                                    ┌──────────────────────┐             │
│                                    │ Concat: [Q, Steps]   │             │
│                                    └───────────┬──────────┘             │
│                                                │                        │
│                                                ▼                        │
│                                    ┌──────────────────────┐             │
│                                    │    LLM (冻结)        │             │
│                                    │   Generate Answer    │             │
│                                    └───────────┬──────────┘             │
│                                                │                        │
│                                                ▼                        │
│                                    ┌──────────────────────┐             │
│                                    │      Answer          │             │
│                                    │      (文本)          │             │
│                                    └──────────────────────┘             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块组成

| 模块 | 类名 | 文件 | 训练状态 |
|------|-----|------|---------|
| 主模型 | `LitDiffLaRFused` | `difflar_fused.py` | - |
| LLM | 预训练模型 | - | **冻结** |
| Embedding Layer | LLM.embed_tokens | - | **冻结** |
| Latent Diffusion | `LatentDiffusion` | `latent_diffusion.py` | **训练** |
| Denoiser | `Denoiser` | `denoiser.py` | **训练** |
| Noise Scheduler | `NoiseScheduler` | `scheduler.py` | 无参数 |

---

## 3. 核心组件详解

### 3.1 Latent Diffusion Model

**位置**：`src/modules/diffusion/latent_diffusion.py`

Latent Diffusion 是整个系统的核心，负责在 LLM 的 Embedding 空间进行扩散和去噪。

#### 3.1.1 初始化参数

```python
LatentDiffusion(
    hidden_size=4096,          # LLM 的隐藏层维度
    num_timesteps=1000,        # 扩散时间步总数
    num_layers=6,              # Denoiser Transformer 层数
    num_heads=8,               # 注意力头数
    max_latent_length=256,     # 最大 latent 序列长度
    schedule_type="linear",    # 噪声调度类型
    normalize_latent=True,     # 是否归一化 latent
    prediction_type="epsilon", # 预测类型：epsilon/x0/flow
    sampler_type="ddim",       # 采样器类型：ddpm/ddim/euler
    ddim_eta=0.0,              # DDIM 随机性（0=确定性）
)
```

#### 3.1.2 归一化机制

Embedding 空间的数值分布与标准正态分布差异较大，直接进行扩散会导致训练-推理分布偏移。归一化机制解决这一问题：

```python
# 归一化公式
x_normalized = (x - mean) / std * scale

# 反归一化公式
x_original = (x_normalized / scale) * std + mean
```

参数 `mean`、`std`、`scale` 在训练开始时从数据估计，并保存到 checkpoint。

#### 3.1.3 支持的扩散模式

| 模式 | prediction_type | 训练目标 | 推理方式 |
|------|----------------|---------|---------|
| DDPM/DDIM | `epsilon` | 预测噪声 ε | 迭代去噪 |
| X0 预测 | `x0` | 直接预测 x₀ | 迭代去噪 |
| Flow Matching | `flow` | 预测速度 v | 欧拉积分 |

### 3.2 Denoiser（去噪网络）

**位置**：`src/modules/diffusion/denoiser.py`

Denoiser 是一个 Transformer 架构的神经网络，负责根据条件预测噪声/x₀/速度。

#### 3.2.1 输入输出

```
输入：
  - x_t: 加噪的 latent [B, L, H]
  - t: 时间步 [B]
  - condition: Query Embedding [B, L_q, H]
  - attention_mask: latent 的有效位置掩码 [B, L]
  - condition_mask: query 的有效位置掩码 [B, L_q]
  - x0_cond: Self-Conditioning 的上一步预测 [B, L, H] (可选)

输出：
  - predicted: 预测的噪声/x₀/速度 [B, L, H]
```

#### 3.2.2 Self-Conditioning

Self-Conditioning 是一种提高生成质量的技术：

```python
# 训练时（50%概率启用）
if use_self_cond:
    with torch.no_grad():
        prev_output = denoiser(x_t, t, condition, x0_cond=None)
        x0_cond = get_x0_from_output(prev_output)
    
    output = denoiser(x_t, t, condition, x0_cond=x0_cond)  # 使用上一步预测作为条件

# 推理时（始终启用）
for t in timesteps:
    x0_cond = x0_pred  # 上一步的预测
    output = denoiser(x_t, t, condition, x0_cond=x0_cond)
    x0_pred = get_x0_from_output(output)  # 保存供下一步使用
```

### 3.3 Noise Scheduler（噪声调度器）

**位置**：`src/modules/diffusion/scheduler.py`

控制扩散过程中噪声的添加和去除。

#### 3.3.1 前向扩散

```python
# 公式：x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
x_t, noise = scheduler.add_noise(x_0, t)
```

#### 3.3.2 逆向去噪（DDPM）

```python
# 公式：x_{t-1} = (x_t - β_t/√(1-ᾱ_t) * ε) / √(α_t) + σ_t * z
x_prev = scheduler.denoise_step(x_t, predicted_noise, t, add_noise=True)
```

#### 3.3.3 DDIM 采样

```python
# 确定性采样（eta=0）或随机采样（eta>0）
x_prev = scheduler.ddim_step(x_t, predicted_noise, t, t_prev, eta)
```

### 3.4 Flow Matching 详解（推荐模式）

**Flow Matching 是一种更先进的扩散训练方法**，使用线性插值路径和速度预测，相比 DDPM/DDIM 具有更稳定的训练和更快的推理。

#### 3.4.1 核心思想对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DDPM/DDIM vs Flow Matching                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DDPM/DDIM（弯曲路径）：                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε                              │   │
│  │                                                                  │   │
│  │  数据 x_0 ──────╮                                                │   │
│  │                  ╲  ← 弯曲路径（需要复杂的 alpha 调度）           │   │
│  │                   ╲                                              │   │
│  │  噪声 ε  ──────────────→ x_t                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Flow Matching（线性路径）：                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  x_t = (1-t) * x_0 + t * ε                                       │   │
│  │                                                                  │   │
│  │  数据 x_0 ────────────────→ x_t ← 线性插值（最简单）             │   │
│  │                ↗                                                 │   │
│  │  噪声 ε  ─────                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 3.4.2 Flow Matching 训练前向过程（逐步详解）

**输入**：
- `steps_embeds`: GT 思考步骤 Embedding [B, L, H]
- `condition`: Query Embedding [B, L_q, H]
- `attention_mask`: 有效位置掩码 [B, L]

**完整步骤**：

```python
def forward_flow_matching(steps_embeds, condition, attention_mask):
    """
    Flow Matching 训练前向传播
    
    位置: src/modules/diffusion/latent_diffusion.py::_forward_flow_matching()
    """
    batch_size = steps_embeds.shape[0]
    device = steps_embeds.device
    
    # ═══════════════════════════════════════════════════════════════
    # Step 1: 归一化（关键！将 Embedding 映射到标准分布）
    # ═══════════════════════════════════════════════════════════════
    # 问题：LLM Embedding 的数值范围与标准正态差异大
    # 解决：归一化到接近 N(0, 1)，使扩散过程更稳定
    # 
    # 公式：x_normalized = (x - mean) / std * scale
    # - mean, std: 从训练数据估计的统计量 [H]
    # - scale: 使范数接近 sqrt(H) 的缩放因子
    
    x_0 = (steps_embeds - latent_mean) / latent_std * latent_scale
    # x_0: [B, L, H]，现在接近标准正态分布
    
    # ═══════════════════════════════════════════════════════════════
    # Step 2: 采样连续时间 t ∈ [0, 1]
    # ═══════════════════════════════════════════════════════════════
    # 注意：Flow Matching 使用连续时间，不是离散时间步
    # t=0 表示纯数据，t=1 表示纯噪声
    
    t = torch.rand(batch_size, device=device)  # [B]
    # 每个样本独立采样一个时间点
    
    # ═══════════════════════════════════════════════════════════════
    # Step 3: 采样噪声（只在有效位置）
    # ═══════════════════════════════════════════════════════════════
    # 关键改进：padding 位置不加噪声
    
    noise = torch.randn_like(x_0)  # [B, L, H]
    
    if attention_mask is not None:
        mask_expanded = attention_mask.unsqueeze(-1)  # [B, L, 1]
        noise = noise * mask_expanded  # padding 位置噪声置零
    
    # ═══════════════════════════════════════════════════════════════
    # Step 4: 线性插值构造 x_t（Flow Matching 核心）
    # ═══════════════════════════════════════════════════════════════
    # 公式：x_t = (1-t) * x_0 + t * noise
    # 
    # 物理意义：
    # - t=0 时：x_t = x_0（纯数据）
    # - t=1 时：x_t = noise（纯噪声）
    # - 0<t<1：数据和噪声的线性混合
    
    t_expand = t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting
    x_t = (1 - t_expand) * x_0 + t_expand * noise
    # x_t: [B, L, H]
    
    # ═══════════════════════════════════════════════════════════════
    # Step 5: 计算目标速度（预测目标）
    # ═══════════════════════════════════════════════════════════════
    # 公式：velocity = noise - x_0
    # 
    # 物理意义：从数据 x_0 到噪声 noise 的方向向量
    # 推理时反向：从噪声沿 -velocity 方向走回数据
    
    target_velocity = noise - x_0  # [B, L, H]
    
    # ═══════════════════════════════════════════════════════════════
    # Step 6: Self-Conditioning（可选，提升质量）
    # ═══════════════════════════════════════════════════════════════
    # 思想：先做一次预测，将预测结果作为额外条件再预测一次
    # 效果：让模型利用自己的预测改进结果
    
    x0_cond = None
    if use_self_cond and random() < 0.5:  # 训练时 50% 概率启用
        with torch.no_grad():
            # 将连续时间转为离散时间步（Denoiser 需要）
            t_discrete = (t * num_timesteps).long().clamp(0, num_timesteps-1)
            
            # 第一次预测
            prev_velocity = denoiser(x_t, t_discrete, condition, 
                                     attention_mask, condition_mask, 
                                     x0_cond=None)
            
            # 从预测速度反推 x_0
            # 由 x_t = (1-t)*x_0 + t*noise 和 v = noise - x_0
            # 推导：x_0 = (x_t - t*v) / (1-t)
            x0_cond = (x_t - t_expand * prev_velocity) / (1 - t_expand + 1e-6)
    
    # ═══════════════════════════════════════════════════════════════
    # Step 7: Denoiser 预测速度
    # ═══════════════════════════════════════════════════════════════
    # Denoiser 输入：
    # - x_t: 加噪的 latent
    # - t_discrete: 离散时间步（用于时间编码）
    # - condition: Query Embedding（条件）
    # - x0_cond: Self-Conditioning 的 x_0 预测（可选）
    
    t_discrete = (t * num_timesteps).long().clamp(0, num_timesteps-1)
    predicted_velocity = denoiser(x_t, t_discrete, condition, 
                                  attention_mask, condition_mask,
                                  x0_cond=x0_cond)
    # predicted_velocity: [B, L, H]
    
    # ═══════════════════════════════════════════════════════════════
    # Step 8: 计算损失（MSE）
    # ═══════════════════════════════════════════════════════════════
    # 只在有效位置（mask=1）计算损失
    
    loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
    loss = loss.mean(dim=-1)  # [B, L]
    
    if attention_mask is not None:
        loss = (loss * attention_mask).sum() / (attention_mask.sum() + 1e-8)
    else:
        loss = loss.mean()
    
    return loss, predicted_velocity, target_velocity
```

#### 3.4.3 Flow Matching 推理过程（逐步详解）

**输入**：
- `condition`: Query Embedding [B, L_q, H]
- `num_inference_steps`: 采样步数（如 50）

**完整步骤**：

```python
def generate_flow_matching(condition, num_inference_steps=50):
    """
    Flow Matching 推理：从噪声生成 Steps Embedding
    
    位置: src/modules/diffusion/latent_diffusion.py::_generate_flow_matching()
    """
    batch_size = condition.shape[0]
    device = condition.device
    
    # ═══════════════════════════════════════════════════════════════
    # Step 1: 从纯噪声开始（t=1）
    # ═══════════════════════════════════════════════════════════════
    x_t = torch.randn(batch_size, max_latent_length, hidden_size, device=device)
    # x_t: [B, 256, 4096]，纯噪声
    
    # ═══════════════════════════════════════════════════════════════
    # Step 2: 设置时间步序列
    # ═══════════════════════════════════════════════════════════════
    # 从 t=1（噪声）积分到 t≈0（数据）
    # 注意：不是到 t=0，而是到 t=dt，避免除零
    
    dt = 1.0 / num_inference_steps  # 步长，例如 1/50 = 0.02
    timesteps = torch.linspace(1.0, dt, num_inference_steps, device=device)
    # timesteps: [1.0, 0.98, 0.96, ..., 0.02]
    
    x0_pred = None  # 用于 Self-Conditioning
    
    # ═══════════════════════════════════════════════════════════════
    # Step 3: 欧拉积分（逐步从噪声走向数据）
    # ═══════════════════════════════════════════════════════════════
    for i, t_val in enumerate(timesteps):
        t = torch.full((batch_size,), t_val, device=device)
        t_discrete = (t * num_timesteps).long().clamp(0, num_timesteps-1)
        
        # ─────────────────────────────────────────────────────────
        # 3.1 Self-Conditioning：使用上一步的 x_0 预测
        # ─────────────────────────────────────────────────────────
        x0_cond = x0_pred  # 第一步为 None，后续步使用上一步预测
        
        # ─────────────────────────────────────────────────────────
        # 3.2 Denoiser 预测速度
        # ─────────────────────────────────────────────────────────
        velocity = denoiser(x_t, t_discrete, condition,
                           attention_mask=None, condition_mask=condition_mask,
                           x0_cond=x0_cond)
        # velocity: [B, L, H]，预测的 noise - x_0 方向
        
        # ─────────────────────────────────────────────────────────
        # 3.3 计算 x_0 预测（供下一步 Self-Conditioning）
        # ─────────────────────────────────────────────────────────
        # 由 x_t = (1-t)*x_0 + t*noise 和 v = noise - x_0
        # 推导：x_0 = (x_t - t*v) / (1-t)
        
        t_expand = t.view(-1, 1, 1)
        x0_pred = (x_t - t_expand * velocity) / (1 - t_expand + 1e-6)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)  # 防止数值爆炸
        
        # ─────────────────────────────────────────────────────────
        # 3.4 欧拉积分更新（核心！）
        # ─────────────────────────────────────────────────────────
        # 公式：x_{t-dt} = x_t - dt * velocity
        # 
        # 物理意义：沿着 -velocity 方向（从噪声到数据）走一小步
        # 每步走 dt 的距离，共走 num_inference_steps 步
        
        x_t = x_t - dt * velocity
        x_t = torch.clamp(x_t, -6.0, 6.0)  # 防止数值爆炸
    
    # ═══════════════════════════════════════════════════════════════
    # Step 4: 反归一化（恢复到原始 Embedding 尺度）
    # ═══════════════════════════════════════════════════════════════
    # 公式：x_original = (x_normalized / scale) * std + mean
    
    x_0 = (x_t / latent_scale) * latent_std + latent_mean
    # x_0: [B, 256, 4096]，生成的 Steps Embedding
    
    return x_0
```

#### 3.4.4 Flow Matching 与 Noise Scheduler 的关系

**关键区别**：Flow Matching **不使用 NoiseScheduler 的 alpha 调度**！

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Noise Scheduler 使用对比                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DDPM/DDIM 模式：                                                       │
│  ├─ 训练：scheduler.add_noise(x_0, t) 使用 sqrt_alpha_cumprod          │
│  ├─ 推理：scheduler.ddim_step() 使用复杂的 alpha 计算                   │
│  └─ 依赖：betas, alphas, alphas_cumprod 等调度参数                      │
│                                                                         │
│  Flow Matching 模式：                                                   │
│  ├─ 训练：直接线性插值 x_t = (1-t)*x_0 + t*noise                        │
│  ├─ 推理：欧拉积分 x_{t-dt} = x_t - dt*velocity                         │
│  └─ 依赖：无！只需要 num_timesteps 用于离散化时间编码                    │
│                                                                         │
│  结论：Flow Matching 更简单，不依赖复杂的噪声调度                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 3.4.5 Flow Matching 优势总结

| 方面 | DDPM/DDIM | Flow Matching |
|------|-----------|---------------|
| 插值公式 | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ | $x_t = (1-t) x_0 + t \epsilon$ |
| 路径形状 | 弯曲（受 alpha schedule 影响） | 线性（最短路径） |
| 预测目标 | 噪声 ε（高方差） | 速度 v = ε - x_0（低方差） |
| 推理方式 | 迭代去噪（复杂公式） | 欧拉积分（简单减法） |
| 数值稳定性 | 依赖 schedule 设计 | 天然稳定 |
| 推理步数 | 通常 50-1000 步 | 10-50 步即可 |
| 单步生成潜力 | 需要蒸馏 | Rectified Flow 可直接支持 |

#### 3.4.6 启用 Flow Matching 配置

```yaml
# configs/difflar_fused.yaml
difflar_config:
  prediction_type: "flow"      # 改为 flow（原 epsilon）
  sampler_type: "euler"        # 改为 euler（原 ddim）
  num_inference_steps: 50      # 可减少（原 128）
```

---

## 4. 训练阶段

### 4.1 两阶段训练策略

```
┌────────────────────────────────────────────────────────────────┐
│                      训练阶段概览                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Stage 1: 基础学习 (Epoch 0 ~ stage1_epochs-1)                 │
│  ├─ Diffusion Loss: 用 GT latent 训练                         │
│  ├─ Self-Conditioning: 50% 概率启用                            │
│  ├─ Answer Loss: 用生成 latent，权重 0.5                       │
│  └─ Alignment Loss: 对齐生成与 GT                              │
│                                                                │
│  ──────────────── 阶段切换 ────────────────                    │
│                                                                │
│  Stage 2: Rollout 强化 (Epoch stage1_epochs ~ max_epochs)      │
│  ├─ Diffusion Loss: 仍用 GT latent（目标不变）                  │
│  ├─ Self-Conditioning: 100% 启用                               │
│  ├─ Rollout: 按概率使用生成 latent 计算 Answer Loss            │
│  ├─ Curriculum: Rollout 比例逐步增加                           │
│  └─ Alignment Loss: 对齐生成与 GT                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Stage 1：基础学习

**目标**：让 Diffusion 模型学会基本的去噪能力

**数据流**：

```python
def forward_stage1(batch):
    # 1. 准备输入
    question = batch["question"]           # 问题文本
    steps = batch["steps"]                 # GT 思考步骤文本
    answer = batch["answer"]               # 答案文本
    
    # 2. 编码 Question
    query_embedding = embedding(tokenizer(question))  # [B, L_q, H]
    
    # 3. 编码 GT Steps（作为扩散目标）
    gt_steps_embeds = embedding(tokenizer(steps))     # [B, L, H]
    gt_steps_embeds = pad_to_fixed_length(gt_steps_embeds, max_length=256)
    
    # 4. Diffusion Loss（用 GT 训练）
    x_0 = normalize(gt_steps_embeds)
    t = random_timestep()
    x_t, noise = add_noise(x_0, t)
    predicted_noise = denoiser(x_t, t, condition=query_embedding)
    diffusion_loss = MSE(predicted_noise, noise)
    
    # 5. 生成 latent（用于 LLM 训练）
    generated_steps_embeds = diffusion.generate(query_embedding)  # [B, L, H]
    
    # 6. Alignment Loss（中间监督）
    alignment_loss = MSE(generated_steps_embeds, gt_steps_embeds) 
                   + 0.1 * CosineDistance(generated_steps_embeds, gt_steps_embeds)
    
    # 7. Answer Loss（LLM 生成答案）
    all_embeds = concat([query_embedding, generated_steps_embeds, answer_embeds])
    answer_loss = LLM.forward(all_embeds, labels=answer_ids).loss
    
    # 8. 总损失
    total_loss = diffusion_loss + alignment_loss + 0.5 * answer_loss
    
    return total_loss
```

### 4.3 Stage 2：Rollout 强化

**目标**：让模型学会在"自身产生误差"时依然能修正

**关键变化**：

1. **Rollout 机制**：按概率使用生成的 latent 而非 GT
2. **Curriculum**：Rollout 比例从 30% 逐步增加到 100%
3. **Self-Conditioning**：100% 启用

```python
def forward_stage2(batch):
    # ... 前面步骤相同 ...
    
    # 4. Diffusion Loss（仍用 GT 作为目标！）
    diffusion_loss = compute_diffusion_loss(gt_steps_embeds, query_embedding)
    
    # 5. Rollout 决策
    rollout_ratio = get_current_rollout_ratio()  # 0.3 → 1.0 递增
    
    if random() < rollout_ratio:
        # 使用生成的 latent
        generated_steps_embeds = diffusion.generate(query_embedding)
        alignment_loss = compute_alignment_loss(generated_steps_embeds, gt_steps_embeds)
    else:
        # 使用 GT（稳定训练）
        generated_steps_embeds = gt_steps_embeds
        alignment_loss = 0.0
    
    # 6. Answer Loss
    answer_loss = LLM.forward([query, generated_steps_embeds, answer]).loss
    
    # 7. 总损失（全权重）
    total_loss = diffusion_loss + alignment_loss + 5.0 * answer_loss
    
    return total_loss
```

### 4.4 Rollout 比例计算

```python
def get_current_rollout_ratio():
    if training_stage == 1:
        return 0.0
    
    stage2_epoch = current_epoch - stage1_epochs
    total_stage2_epochs = max_epochs - stage1_epochs
    
    progress = stage2_epoch / total_stage2_epochs  # 0.0 → 1.0
    ratio = rollout_start_ratio + (rollout_final_ratio - rollout_start_ratio) * progress
    
    return ratio  # 例如：0.3 → 0.5 → 0.7 → 1.0
```

---

## 5. 推理阶段

### 5.1 推理流程

```python
@torch.no_grad()
def latent_generate(questions: List[str]) -> List[str]:
    # 1. 编码 Question
    query_embedding = embedding(tokenizer(questions))  # [B, L_q, H]
    
    # 2. Diffusion 生成 Steps Embedding
    if use_cfg:
        steps_embeds = diffusion.generate_with_cfg(
            condition=query_embedding,
            num_inference_steps=128,
            cfg_scale=1.5,
        )
    else:
        steps_embeds = diffusion.generate(
            condition=query_embedding,
            num_inference_steps=128,
        )
    # steps_embeds: [B, 256, H]
    
    # 3. 拼接并生成 Answer
    all_embeds = concat([query_embedding, steps_embeds, separator_embeds])
    
    # 4. LLM 自回归生成答案
    answer_ids = LLM.generate(
        inputs_embeds=all_embeds,
        max_new_tokens=32,
        do_sample=False,
    )
    
    # 5. 解码
    answers = tokenizer.decode(answer_ids)
    return answers
```

### 5.2 Diffusion 采样过程

#### 5.2.1 DDIM 采样

```python
def generate_ddim(condition, num_steps=128):
    # 1. 从纯噪声开始
    x_t = randn(batch_size, latent_length, hidden_size)
    
    # 2. 获取时间步序列（均匀间隔）
    timesteps = linspace(999, 0, num_steps)  # [999, 991, 983, ...]
    
    x0_pred = None  # Self-Conditioning
    
    # 3. 迭代去噪
    for i, t in enumerate(timesteps):
        # Self-Conditioning：使用上一步预测
        x0_cond = x0_pred
        
        # 预测噪声
        predicted_noise = denoiser(x_t, t, condition, x0_cond=x0_cond)
        
        # 更新 x0_pred（供下一步 Self-Conditioning）
        x0_pred = get_x0_from_noise(x_t, predicted_noise, t)
        x0_pred = clamp(x0_pred, -3.0, 3.0)  # 防止数值爆炸
        
        # DDIM 更新步
        t_prev = timesteps[i+1] if i+1 < len(timesteps) else -1
        x_t = ddim_step(x_t, predicted_noise, t, t_prev, eta=0.0)
    
    # 4. 反归一化
    x_0 = denormalize(x_t)
    return x_0
```

#### 5.2.2 Flow Matching 采样

```python
def generate_flow_matching(condition, num_steps=128):
    # 1. 从纯噪声开始（t=1）
    x_t = randn(batch_size, latent_length, hidden_size)
    
    # 2. 时间步：从 1 → 0
    dt = 1.0 / num_steps
    timesteps = linspace(1.0, dt, num_steps)  # [1.0, 0.992, 0.984, ...]
    
    x0_pred = None
    
    # 3. 欧拉积分
    for t in timesteps:
        # 预测速度
        velocity = denoiser(x_t, t, condition, x0_cond=x0_pred)
        
        # 计算 x0_pred（供 Self-Conditioning）
        # 公式：x_t = (1-t)*x_0 + t*noise，v = noise - x_0
        # 推导：x_0 = (x_t - t*v) / (1-t)
        x0_pred = (x_t - t * velocity) / (1 - t + 1e-6)
        x0_pred = clamp(x0_pred, -3.0, 3.0)
        
        # 欧拉积分：向数据方向移动
        x_t = x_t - dt * velocity
    
    # 4. 反归一化
    x_0 = denormalize(x_t)
    return x_0
```

### 5.3 Classifier-Free Guidance (CFG)

CFG 通过放大条件信号来提高生成质量：

```python
def generate_with_cfg(condition, cfg_scale=1.5):
    uncond = zeros_like(condition)  # 无条件向量
    
    for t in timesteps:
        # 有条件预测
        output_cond = denoiser(x_t, t, condition)
        
        # 无条件预测
        output_uncond = denoiser(x_t, t, uncond)
        
        # CFG 组合
        output = output_uncond + cfg_scale * (output_cond - output_uncond)
        
        # 更新 x_t
        x_t = update_step(x_t, output, t)
    
    return x_t
```

---

## 6. 数据流详解

### 6.1 训练时数据流

```
输入 Batch:
{
    "question": ["小明有5个苹果...", "计算 3+4=?", ...],  # [B]
    "steps": ["Step 1: 5个苹果...", "Step 1: 3+4...", ...],  # [B]
    "answer": ["8", "7", ...],  # [B]
}

                    ▼

┌─────────────────────────────────────────────────────────────┐
│                     Tokenize & Embed                         │
└─────────────────────────────────────────────────────────────┘
                    ▼

query_embedding: [B, L_q, H]     # 例如 [8, 128, 4096]
gt_steps_embeds: [B, L, H]       # 例如 [8, 256, 4096]
answer_embeds: [B, L_a, H]       # 例如 [8, 16, 4096]

                    ▼

┌─────────────────────────────────────────────────────────────┐
│                   Latent Diffusion Forward                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ x_0 = normalize(gt_steps_embeds)                    │    │
│  │ t = randint(0, 1000)                                │    │
│  │ x_t, noise = add_noise(x_0, t)                      │    │
│  │ predicted = denoiser(x_t, t, query_embedding)       │    │
│  │ diffusion_loss = MSE(predicted, noise)              │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                    ▼

┌─────────────────────────────────────────────────────────────┐
│                   Generate for Training                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ generated_steps_embeds = diffusion.generate(        │    │
│  │     condition=query_embedding,                      │    │
│  │     num_inference_steps=10,  # 训练时用少量步数     │    │
│  │     enable_grad=True,        # 保留梯度             │    │
│  │ )                                                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                    ▼

generated_steps_embeds: [B, L, H]   # [8, 256, 4096]

                    ▼

┌─────────────────────────────────────────────────────────────┐
│                   Alignment Loss                             │
│  alignment_loss = MSE(generated, gt) + 0.1 * Cosine(gen, gt)│
└─────────────────────────────────────────────────────────────┘

                    ▼

┌─────────────────────────────────────────────────────────────┐
│                   LLM Forward for Answer Loss                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ all_embeds = concat([query, generated_steps, answer])│    │
│  │ # all_embeds: [B, L_q + L + L_a, H]                 │    │
│  │                                                      │    │
│  │ labels = [-100, ..., -100, answer_ids]              │    │
│  │ # 只计算 answer 部分的 loss                          │    │
│  │                                                      │    │
│  │ answer_loss = LLM.forward(all_embeds, labels).loss  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

                    ▼

total_loss = α * diffusion_loss + β * alignment_loss + γ * answer_loss

                    ▼

┌─────────────────────────────────────────────────────────────┐
│                   Backward & Optimize                        │
│  total_loss.backward()                                       │
│  optimizer.step()                                            │
│  # 注意：只有 Diffusion Model 参数更新，LLM 冻结            │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 推理时数据流

```
输入:
  question: "小明有5个苹果，小红给了他3个，现在有多少？"

                    ▼

┌─────────────────────────────────────────────────────────────┐
│                   Tokenize & Embed                           │
│  query_embedding = embedding(tokenizer(question))            │
│  # [1, 64, 4096]                                             │
└─────────────────────────────────────────────────────────────┘

                    ▼

┌─────────────────────────────────────────────────────────────┐
│                   Diffusion Generate                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ x_t = randn(1, 256, 4096)  # 从噪声开始             │    │
│  │                                                      │    │
│  │ for t in [999, 991, 983, ..., 7, 0]:  # 128步       │    │
│  │     predicted = denoiser(x_t, t, query_embedding)   │    │
│  │     x_t = ddim_step(x_t, predicted, t)              │    │
│  │                                                      │    │
│  │ steps_embeds = denormalize(x_t)                     │    │
│  │ # [1, 256, 4096]                                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

                    ▼

┌─────────────────────────────────────────────────────────────┐
│                   Concat & LLM Generate                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ all_embeds = concat([query_embedding, steps_embeds, │    │
│  │                      separator_embeds])             │    │
│  │ # [1, 64 + 256 + 4, 4096]                           │    │
│  │                                                      │    │
│  │ answer_ids = LLM.generate(                          │    │
│  │     inputs_embeds=all_embeds,                       │    │
│  │     max_new_tokens=32,                              │    │
│  │ )                                                   │    │
│  │ # [1, 32]                                           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

                    ▼

输出:
  answer: "8"
```

---

## 7. 关键技术

### 7.1 Embedding 归一化

**问题**：LLM Embedding 的分布与标准正态差异大

**解决**：在扩散前归一化，推理后反归一化

```python
# 训练开始时估计参数
mean = embeddings.mean(dim=0)           # [H]
std = embeddings.std(dim=0)             # [H]
scale = target_norm / current_norm      # 标量

# 归一化（扩散前）
x_normalized = (x - mean) / std * scale

# 反归一化（推理后）
x_original = (x_normalized / scale) * std + mean
```

### 7.2 变长序列处理

**问题**：不同样本的思考步骤长度不同

**解决**：统一 padding 到固定长度，使用 attention mask

```python
# Padding 策略
if current_length < max_length:
    pad_embeds = PAD_TOKEN_EMBEDDING.expand(batch_size, pad_length, H)
    steps_embeds = concat([steps_embeds, pad_embeds])
    mask = concat([ones(current_length), zeros(pad_length)])

# 扩散时只在有效位置加噪声
noise = randn_like(x_0)
noise = noise * mask.unsqueeze(-1)  # padding 位置噪声为 0

# 损失只在有效位置计算
loss = MSE(predicted, target)
loss = (loss * mask).sum() / mask.sum()
```

### 7.3 Exposure Bias 修复

**问题**：训练用 GT embedding，推理用生成 embedding，分布不匹配

**解决**：训练时也使用生成的 embedding

```python
# 原来（有问题）
generated_steps_embeds = gt_steps_embeds  # 训练用 GT

# 现在（修复后）
generated_steps_embeds = diffusion.generate(query_embedding)  # 训练也用生成
alignment_loss = compute_alignment_loss(generated, gt)  # 添加对齐损失
```

### 7.4 Mask Dropout

**问题**：训练时用真实 mask，推理时用全 1 mask，不一致

**解决**：Stage 2 随机使用全 1 mask

```python
if training_stage == 2:
    if random() < 0.5:
        answer_steps_mask = ones(batch_size, max_length)  # 全 1
    else:
        answer_steps_mask = steps_mask  # 真实 mask
```

---

## 8. 配置参数

### 8.1 DiffLaR Config

```yaml
difflar_config:
  # ═══ 扩散模型 ═══
  num_timesteps: 1000           # 扩散时间步总数
  max_latent_length: 256        # 最大 latent 长度
  denoiser_layers: 6            # Denoiser 层数
  denoiser_heads: 8             # 注意力头数
  dropout: 0.1                  # Dropout 率
  noise_schedule: "linear"      # 噪声调度：linear/cosine/sqrt
  normalize_latent: true        # 是否归一化 latent
  prediction_type: "epsilon"    # 预测类型：epsilon/x0/flow
  sampler_type: "ddim"          # 采样器：ddpm/ddim/euler
  ddim_eta: 0.0                 # DDIM 随机性
  
  # ═══ 推理 ═══
  num_inference_steps: 128      # 推理采样步数
  train_inference_steps: 10     # 训练时生成步数
  use_cfg: false                # 是否使用 CFG
  cfg_scale: 1.5                # CFG 强度
  clamp_value: 3.0              # 数值截断阈值
  
  # ═══ 两阶段训练 ═══
  stage1_epochs: 5              # Stage 1 epoch 数
  rollout_start_ratio: 0.3      # Rollout 起始比例
  rollout_final_ratio: 1.0      # Rollout 最终比例
  rollout_inference_steps: 10   # Rollout 时生成步数
  
  # ═══ Self-Conditioning ═══
  stage1_self_cond_prob: 0.5    # Stage 1 Self-Cond 概率
  stage2_self_cond_prob: 1.0    # Stage 2 Self-Cond 概率
  
  # ═══ 损失权重 ═══
  diffusion_loss_weight: 1.0    # 扩散损失权重
  answer_loss_weight: 5.0       # 答案损失权重
  alignment_loss_weight: 1.0    # 对齐损失权重
```

---

## 9. 损失函数

### 9.1 损失组成

```
Total Loss = α × Diffusion Loss + β × Alignment Loss + γ × Answer Loss
```

### 9.2 各损失详解

#### 9.2.1 Diffusion Loss

**目的**：训练 Denoiser 学会去噪

**公式**（epsilon 预测模式）：

$$\mathcal{L}_{diffusion} = \mathbb{E}_{t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$

其中：
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
- $\epsilon \sim \mathcal{N}(0, I)$
- $c$ 是条件（Query Embedding）

#### 9.2.2 Alignment Loss

**目的**：显式对齐生成 latent 与 GT latent

**公式**：

$$\mathcal{L}_{align} = \text{MSE}(z_{gen}, z_{gt}) + 0.1 \times (1 - \cos(z_{gen}, z_{gt}))$$

其中只在有效位置（mask=1）计算。

#### 9.2.3 Answer Loss

**目的**：确保 LLM 能基于生成的 latent 生成正确答案

**公式**：

$$\mathcal{L}_{answer} = -\sum_{i} \log P(a_i | q, z_{gen}, a_{<i})$$

即标准的交叉熵损失，只在 answer token 位置计算。

### 9.3 损失权重策略

| 阶段 | Diffusion | Alignment | Answer |
|------|-----------|-----------|--------|
| Stage 1 | 1.0 | 1.0 | 0.5 |
| Stage 2 | 1.0 | 1.0 | 5.0 |

---

## 附录 A：代码文件结构

```
src/
├── models/
│   ├── difflar_fused.py      # 主模型类
│   └── model_base.py         # 基类
├── modules/
│   └── diffusion/
│       ├── latent_diffusion.py   # Latent Diffusion 模型
│       ├── denoiser.py           # Denoiser 网络
│       └── scheduler.py          # 噪声调度器
└── utils/
    └── utils.py              # 工具函数
```

---

## 附录 B：关键公式汇总

### 前向扩散

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

### DDPM 去噪

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta \right) + \sigma_t z$$

### DDIM 去噪

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma^2} \epsilon_\theta + \sigma \epsilon$$

### Flow Matching

$$\frac{dx}{dt} = v_\theta(x_t, t, c)$$

$$x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta$$

### Self-Conditioning

$$\hat{x}_0^{(k)} = f_\theta(x_t, t, c, \hat{x}_0^{(k-1)})$$

---

*文档结束*
