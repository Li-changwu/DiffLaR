# DiffLaR Fused 算法详细分析报告

## 一、项目概述

### 1.1 项目背景
DiffLaR Fused 是 CoLaR（Compressed Latent Reasoning）项目中的一个变体实现，旨在通过**扩散模型（Diffusion Model）**在 LLM 的 Embedding 空间中并行生成思考过程的隐表示，从而加速思维链（Chain-of-Thought）推理。

### 1.2 核心目标
- **加速推理**：将自回归的 token-by-token 生成转换为并行的扩散采样
- **保持质量**：在加速的同时保持推理质量
- **动态调整**：支持在推理时动态调整生成速度

### 1.3 关键创新点
1. **Latent Diffusion**：在 Embedding 空间而非 token 空间进行扩散
2. **两阶段训练**：Stage 1 基础学习 + Stage 2 Rollout 强化
3. **Self-Conditioning**：利用上一步预测改进当前预测
4. **Flow Matching**：使用线性插值路径，更稳定的训练和推理
5. **中间监督**：Alignment Loss 显式对齐生成与 GT

---

## 二、算法架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    DiffLaR Fused 系统架构                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Question (文本)                                             │
│       │                                                      │
│       ▼                                                      │
│  Tokenizer + Embedding (冻结)                                │
│       │                                                      │
│       ▼                                                      │
│  Query Embedding [B, L_q, H]                                 │
│       │                                                      │
│       ├──────────────────────────────────────┐              │
│       │                                      │              │
│       ▼                                      ▼              │
│  Latent Diffusion Model              GT Steps Embedding     │
│  ┌────────────────────┐            [B, L, H]                │
│  │  Denoiser          │                                      │
│  │  (Transformer)     │                                      │
│  │  ┌──────────────┐  │                                      │
│  │  │ Self-Attn    │  │                                      │
│  │  │ Cross-Attn   │  │                                      │
│  │  │ Time Embed   │  │                                      │
│  │  └──────────────┘  │                                      │
│  └────────────────────┘                                      │
│       │                                                      │
│       ▼                                                      │
│  Generated Steps Embedding [B, L, H]                        │
│       │                                                      │
│       ├──────────────────┐                                  │
│       │                  │                                  │
│       ▼                  ▼                                  │
│  Alignment Loss    Concat: [Q, Steps, Answer]               │
│       │                  │                                  │
│       │                  ▼                                  │
│       │            LLM (冻结)                                │
│       │                  │                                  │
│       │                  ▼                                  │
│       │            Answer Loss                               │
│       │                  │                                  │
│       └──────────────────┘                                  │
│                  │                                          │
│                  ▼                                          │
│            Total Loss                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块组成

| 模块 | 类名 | 文件路径 | 训练状态 | 功能 |
|------|------|---------|---------|------|
| 主模型 | `LitDiffLaRFused` | `src/models/difflar_fused.py` | - | 协调各模块，实现两阶段训练 |
| LLM | `LlamaForCausalLM` | transformers | **冻结** | 生成最终答案 |
| Embedding | `nn.Embedding` | LLM内置 | **冻结** | Token → Embedding |
| Latent Diffusion | `LatentDiffusion` | `src/modules/diffusion/latent_diffusion.py` | **训练** | 扩散模型核心 |
| Denoiser | `Denoiser` | `src/modules/diffusion/denoiser.py` | **训练** | 去噪网络 |
| Noise Scheduler | `NoiseScheduler` | `src/modules/diffusion/scheduler.py` | 无参数 | 噪声调度 |

---

## 三、核心算法详解

### 3.1 Latent Diffusion 模型

#### 3.1.1 核心思想
在 LLM 的 Embedding 空间（而非 token 空间）进行扩散，生成思考步骤的隐表示。

**优势**：
- Embedding 空间是连续的，适合扩散模型
- 可以并行生成整个序列，而非逐 token 生成
- 生成的 embedding 可以直接输入 LLM

#### 3.1.2 归一化机制（关键！）

**问题**：LLM Embedding 的数值分布与标准正态分布差异很大，直接扩散会导致训练-推理分布偏移。

**解决方案**：数据归一化

```python
# 训练开始时估计参数（从训练数据）
mean = embeddings.mean(dim=0)           # [H]
std = embeddings.std(dim=0)             # [H]
scale = target_norm / current_norm      # 标量

# 归一化（扩散前）
x_normalized = (x - mean) / std * scale

# 反归一化（推理后）
x_original = (x_normalized / scale) * std + mean
```

**实现位置**：
- 估计：`LatentDiffusion.estimate_latent_stats()`
- 归一化：`LatentDiffusion.normalize()`
- 反归一化：`LatentDiffusion.denormalize()`

#### 3.1.3 支持的扩散模式

| 模式 | prediction_type | 训练目标 | 推理方式 | 优势 |
|------|----------------|---------|---------|------|
| DDPM/DDIM | `epsilon` | 预测噪声 ε | 迭代去噪 | 经典方法，稳定 |
| X0 预测 | `x0` | 直接预测 x₀ | 迭代去噪 | 更直观 |
| **Flow Matching** | `flow` | 预测速度 v | **欧拉积分** | **更稳定，更快** |

**推荐使用 Flow Matching**（`prediction_type="flow"`），原因：
- 线性插值路径，避免复杂的 alpha 调度
- 速度预测比噪声预测方差更小
- 推理步数可以更少（10-50 步即可）

---

### 3.2 Flow Matching 详解（推荐模式）

#### 3.2.1 核心公式对比

**DDPM/DDIM（弯曲路径）**：
```
x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
```
需要复杂的 alpha 调度，路径是弯曲的。

**Flow Matching（线性路径）**：
```
x_t = (1-t) * x_0 + t * ε
```
最简单的线性插值，t ∈ [0, 1]。

#### 3.2.2 训练过程（逐步详解）

**位置**：`LatentDiffusion._forward_flow_matching()`

```python
def _forward_flow_matching(x_0, condition, attention_mask):
    """
    输入：
    - x_0: GT Steps Embedding [B, L, H]（已归一化）
    - condition: Query Embedding [B, L_q, H]
    - attention_mask: 有效位置掩码 [B, L]
    """
    
    # Step 1: 采样连续时间 t ~ U[0, 1]
    t = torch.rand(batch_size, device=device)  # [B]
    
    # Step 2: 采样噪声（关键：只在有效位置加噪声）
    noise = torch.randn_like(x_0)  # [B, L, H]
    if attention_mask is not None:
        mask_expanded = attention_mask.unsqueeze(-1)  # [B, L, 1]
        noise = noise * mask_expanded  # padding位置噪声为0
    
    # Step 3: 线性插值构造 x_t
    t_expand = t.view(-1, 1, 1)  # [B, 1, 1]
    x_t = (1 - t_expand) * x_0 + t_expand * noise
    
    # Step 4: 计算目标速度
    # 速度 = 从数据指向噪声的方向向量
    target_velocity = noise - x_0  # [B, L, H]
    
    # Step 5: Self-Conditioning（可选）
    x0_cond = None
    if use_self_cond and random() < 0.5:
        with torch.no_grad():
            # 第一次预测
            t_discrete = (t * num_timesteps).long()
            prev_velocity = denoiser(x_t, t_discrete, condition, x0_cond=None)
            # 从速度反推 x_0
            x0_cond = (x_t - t_expand * prev_velocity) / (1 - t_expand + 1e-6)
    
    # Step 6: Denoiser 预测速度
    t_discrete = (t * num_timesteps).long()
    predicted_velocity = denoiser(x_t, t_discrete, condition, 
                                   attention_mask, condition_mask,
                                   x0_cond=x0_cond)
    
    # Step 7: 计算损失（MSE，只在有效位置）
    loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
    loss = loss.mean(dim=-1)  # [B, L]
    if attention_mask is not None:
        loss = (loss * attention_mask).sum() / (attention_mask.sum() + 1e-8)
    else:
        loss = loss.mean()
    
    return loss, predicted_velocity, target_velocity
```

#### 3.2.3 推理过程（逐步详解）

**位置**：`LatentDiffusion._generate_flow_matching()`

```python
def _generate_flow_matching(condition, num_inference_steps=50):
    """
    从噪声生成 Steps Embedding
    """
    
    # Step 1: 从纯噪声开始（t=1）
    x_t = torch.randn(batch_size, max_latent_length, hidden_size, device=device)
    
    # Step 2: 设置时间步序列（从 t=1 到 t≈0）
    dt = 1.0 / num_inference_steps  # 步长，例如 1/50 = 0.02
    timesteps = torch.linspace(1.0, dt, num_inference_steps, device=device)
    # timesteps: [1.0, 0.98, 0.96, ..., 0.02]
    
    x0_pred = None  # 用于 Self-Conditioning
    
    # Step 3: 欧拉积分（逐步从噪声走向数据）
    for i, t_val in enumerate(timesteps):
        t = torch.full((batch_size,), t_val, device=device)
        t_discrete = (t * num_timesteps).long()
        
        # 3.1 Self-Conditioning：使用上一步的 x_0 预测
        x0_cond = x0_pred if use_self_cond else None
        
        # 3.2 Denoiser 预测速度
        velocity = denoiser(x_t, t_discrete, condition,
                           attention_mask=None, condition_mask=condition_mask,
                           x0_cond=x0_cond)
        
        # 3.3 计算 x_0 预测（供下一步 Self-Conditioning）
        t_expand = t.view(-1, 1, 1)
        x0_pred = (x_t - t_expand * velocity) / (1 - t_expand + 1e-6)
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
        
        # 3.4 欧拉积分更新（核心！）
        # 公式：x_{t-dt} = x_t - dt * velocity
        # 物理意义：沿着 -velocity 方向（从噪声到数据）走一小步
        x_t = x_t - dt * velocity
        x_t = torch.clamp(x_t, -6.0, 6.0)
    
    # Step 4: 反归一化
    x_0 = denormalize(x_t)
    return x_0
```

**关键点**：
- 欧拉积分：`x_{t-dt} = x_t - dt * velocity`
- 从 t=1（噪声）积分到 t≈0（数据）
- 每步走 `dt` 的距离，共走 `num_inference_steps` 步

---

### 3.3 Denoiser（去噪网络）

#### 3.3.1 架构设计

**位置**：`src/modules/diffusion/denoiser.py`

```
输入: x_t [B, L, H] + t [B] + condition [B, L_q, H] + x0_cond [B, L, H]
       │
       ▼
┌──────────────────────┐
│ Time Embedding       │  ← 时间步编码
│ (Sinusoidal + MLP)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ AdaLN Modulation     │  ← 自适应 LayerNorm 参数
│ (时间步 → shift/scale)│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Self-Conditioning    │
│ x0_proj(x0_cond)     │  ← 上一步预测的 x0
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Input Projection     │
│ input_proj(x_t) + x0 │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Position Embedding   │
│ (序列位置编码)         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Cross-Attention      │  ← 融合 Query 条件
│ (x_t → condition)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Transformer Encoder  │  ← Self-Attention
│ (num_layers 层)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Output Projection    │
│ (预测噪声/速度)       │
└──────────┬───────────┘
           │
           ▼
输出: predicted [B, L, H]
```

#### 3.3.2 Self-Conditioning 机制

**核心思想**：将上一步预测的 x₀ 作为额外条件输入，让模型利用自己的预测改进结果。

**实现**：
```python
# 训练时（50%概率启用）
if use_self_cond and random() < 0.5:
    with torch.no_grad():
        prev_output = denoiser(x_t, t, condition, x0_cond=None)
        x0_cond = get_x0_from_output(prev_output)
    
    output = denoiser(x_t, t, condition, x0_cond=x0_cond)  # 使用上一步预测

# 推理时（始终启用）
for t in timesteps:
    x0_cond = x0_pred  # 上一步的预测
    output = denoiser(x_t, t, condition, x0_cond=x0_cond)
    x0_pred = get_x0_from_output(output)  # 保存供下一步使用
```

**效果**：提高生成质量，减少误差累积。

---

### 3.4 两阶段训练策略

#### 3.4.1 训练阶段概览

```
┌─────────────────────────────────────────────────────────────┐
│                    两阶段训练策略                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: 基础学习 (Epoch 0 ~ stage1_epochs-1)              │
│  ├─ Diffusion Loss: 用 GT latent 训练                        │
│  ├─ Self-Conditioning: 50% 概率启用                          │
│  ├─ Answer Loss: 用生成 latent，权重 0.5                     │
│  └─ Alignment Loss: 对齐生成与 GT                            │
│                                                              │
│  ──────────────── 阶段切换 ────────────────                  │
│                                                              │
│  Stage 2: Rollout 强化 (Epoch stage1_epochs ~ max_epochs)   │
│  ├─ Diffusion Loss: 仍用 GT latent（目标不变）               │
│  ├─ Self-Conditioning: 100% 启用                             │
│  ├─ Rollout: 按概率使用生成 latent 计算 Answer Loss          │
│  ├─ Curriculum: Rollout 比例逐步增加 (0.3 → 1.0)            │
│  └─ Alignment Loss: 对齐生成与 GT                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 3.4.2 Stage 1：基础学习

**目标**：让 Diffusion 模型学会基本的去噪能力

**数据流**：
```python
def forward_stage1(batch):
    # 1. 准备输入
    question = batch["question"]
    steps = batch["steps"]      # GT 思考步骤
    answer = batch["answer"]
    
    # 2. 编码 Question
    query_embedding = embedding(tokenizer(question))
    
    # 3. 编码 GT Steps
    gt_steps_embeds = embedding(tokenizer(steps))
    gt_steps_embeds = pad_to_fixed_length(gt_steps_embeds, max_length=256)
    
    # 4. Diffusion Loss（用 GT 训练）
    diffusion_loss = compute_diffusion_loss(gt_steps_embeds, query_embedding)
    
    # 5. 生成 latent（用于 LLM 训练）
    generated_steps_embeds = diffusion.generate(query_embedding, num_steps=10)
    
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

**关键点**：
- Diffusion Loss 用 GT 作为目标
- Answer Loss 用生成的 latent（避免 Exposure Bias）
- Alignment Loss 显式对齐生成与 GT

#### 3.4.3 Stage 2：Rollout 强化

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

**Rollout 比例计算**：
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

### 3.5 损失函数

#### 3.5.1 损失组成

```
Total Loss = α × Diffusion Loss + β × Alignment Loss + γ × Answer Loss
```

#### 3.5.2 各损失详解

**1. Diffusion Loss**
- **目的**：训练 Denoiser 学会去噪
- **公式**（Flow Matching）：
  ```
  L_diffusion = E_t [|| velocity - velocity_θ(x_t, t, c) ||²]
  ```
  其中 `velocity = noise - x_0`

**2. Alignment Loss**
- **目的**：显式对齐生成 latent 与 GT latent
- **公式**：
  ```
  L_align = MSE(z_gen, z_gt) + 0.1 × (1 - cos(z_gen, z_gt))
  ```
  只在有效位置（mask=1）计算

**3. Answer Loss**
- **目的**：确保 LLM 能基于生成的 latent 生成正确答案
- **公式**：
  ```
  L_answer = -Σ_i log P(a_i | q, z_gen, a_{<i})
  ```
  标准的交叉熵损失，只在 answer token 位置计算

#### 3.5.3 损失权重策略

| 阶段 | Diffusion | Alignment | Answer |
|------|-----------|-----------|--------|
| Stage 1 | 1.0 | 1.0 | 0.5 |
| Stage 2 | 1.0 | 1.0 | 5.0 |

**原因**：
- Stage 1：Answer Loss 权重较低，重点学习扩散
- Stage 2：Answer Loss 权重提高，强化端到端性能

---

## 四、关键技术细节

### 4.1 变长序列处理

**问题**：不同样本的思考步骤长度不同

**解决方案**：
1. **统一 Padding**：所有序列 padding 到固定长度（`max_latent_length=256`）
2. **使用 PAD Token Embedding**：而非零向量（零向量在 embedding 空间中不是"无意义"）
3. **Mask 机制**：
   - 只在有效位置加噪声
   - 只在有效位置计算损失

**实现**：
```python
# Padding 策略
if current_length < max_length:
    pad_token_embed = embedding(pad_token_id)
    pad_embeds = pad_token_embed.expand(batch_size, pad_length, H)
    steps_embeds = concat([steps_embeds, pad_embeds])
    mask = concat([ones(current_length), zeros(pad_length)])

# 扩散时只在有效位置加噪声
noise = randn_like(x_0)
noise = noise * mask.unsqueeze(-1)  # padding位置噪声为0

# 损失只在有效位置计算
loss = MSE(predicted, target)
loss = (loss * mask).sum() / mask.sum()
```

### 4.2 Exposure Bias 修复

**问题**：训练用 GT embedding，推理用生成 embedding，分布不匹配

**解决方案**：训练时也使用生成的 embedding

```python
# 原来（有问题）
generated_steps_embeds = gt_steps_embeds  # 训练用 GT

# 现在（修复后）
generated_steps_embeds = diffusion.generate(query_embedding)  # 训练也用生成
alignment_loss = compute_alignment_loss(generated, gt)  # 添加对齐损失
```

### 4.3 Mask Dropout

**问题**：训练时用真实 mask，推理时用全 1 mask，不一致

**解决方案**：Stage 2 随机使用全 1 mask

```python
if training_stage == 2:
    if random() < 0.5:
        answer_steps_mask = ones(batch_size, max_length)  # 全 1
    else:
        answer_steps_mask = steps_mask  # 真实 mask
```

---

## 五、推理流程

### 5.1 完整推理流程

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

### 5.2 推理时间步数

- **训练时**：`train_inference_steps=10`（快速生成，保留梯度）
- **推理时**：`num_inference_steps=128`（高质量生成）
- **Flow Matching**：可以更少（10-50 步即可）

---

## 六、配置参数

### 6.1 关键配置项

```yaml
difflar_config:
  # ═══ 扩散模型 ═══
  num_timesteps: 1000           # 扩散时间步总数
  max_latent_length: 256        # 最大 latent 长度
  denoiser_layers: 6            # Denoiser 层数
  denoiser_heads: 8             # 注意力头数
  dropout: 0.1                  # Dropout 率
  noise_schedule: "linear"      # 噪声调度：linear/cosine
  normalize_latent: true        # 是否归一化 latent
  prediction_type: "flow"       # 预测类型：epsilon/x0/flow（推荐flow）
  sampler_type: "euler"         # 采样器：ddpm/ddim/euler（flow用euler）
  
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
  rollout_inference_steps: 10  # Rollout 时生成步数
  
  # ═══ Self-Conditioning ═══
  stage1_self_cond_prob: 0.5    # Stage 1 Self-Cond 概率
  stage2_self_cond_prob: 1.0   # Stage 2 Self-Cond 概率
  
  # ═══ 损失权重 ═══
  diffusion_loss_weight: 5.0    # 扩散损失权重
  answer_loss_weight: 1.0       # 答案损失权重
  alignment_loss_weight: 1.0    # 对齐损失权重
```

---

## 七、代码文件结构

```
src/
├── models/
│   ├── difflar_fused.py      # 主模型类 LitDiffLaRFused
│   └── model_base.py         # 基类 LitCoTModelBase
├── modules/
│   └── diffusion/
│       ├── latent_diffusion.py   # Latent Diffusion 模型
│       ├── denoiser.py           # Denoiser 网络
│       └── scheduler.py          # 噪声调度器
└── configs/
    └── models/
        └── difflar_fused.yaml    # 配置文件
```

---

## 八、算法优势总结

### 8.1 相比传统 CoT 的优势

1. **加速推理**：并行生成思考步骤，而非逐 token 生成
2. **动态调整**：可以通过调整 `num_inference_steps` 控制速度和质量
3. **保持质量**：通过两阶段训练和中间监督保持推理质量

### 8.2 相比其他 Latent 方法的优势

1. **Flow Matching**：更稳定的训练和更快的推理
2. **Self-Conditioning**：提高生成质量
3. **Rollout Training**：减少 Exposure Bias
4. **中间监督**：Alignment Loss 显式对齐生成与 GT

---

## 九、关键公式汇总

### 9.1 Flow Matching

**训练**：
- 插值：`x_t = (1-t) * x_0 + t * noise`
- 目标速度：`velocity = noise - x_0`
- 损失：`L = || velocity - velocity_θ(x_t, t, c) ||²`

**推理**：
- 欧拉积分：`x_{t-dt} = x_t - dt * velocity`
- 从 t=1（噪声）积分到 t≈0（数据）

### 9.2 Self-Conditioning

- 训练：`x0_cond = get_x0_from_prev_prediction()`
- 推理：`x0_cond = x0_pred`（上一步预测）

### 9.3 归一化

- 归一化：`x_norm = (x - mean) / std * scale`
- 反归一化：`x_orig = (x_norm / scale) * std + mean`

---

## 十、总结

DiffLaR Fused 是一个创新的思维链加速算法，通过以下关键技术实现了高效且高质量的推理：

1. **Latent Diffusion**：在 Embedding 空间进行扩散
2. **Flow Matching**：线性插值路径，更稳定的训练和推理
3. **两阶段训练**：基础学习 + Rollout 强化
4. **Self-Conditioning**：利用上一步预测改进当前预测
5. **中间监督**：Alignment Loss 显式对齐生成与 GT

该算法在保持推理质量的同时，显著加速了思维链推理过程，是一个值得深入研究和应用的方法。

---

*报告生成时间：2026-01-09*
*分析基于代码版本：最新版本*

