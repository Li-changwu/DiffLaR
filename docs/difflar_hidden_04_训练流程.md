# DiffLaR Hidden 训练流程

## 一、训练阶段概览

DiffLaR Hidden 采用**两阶段训练策略**：

1. **Stage1**：Diffusion 模型独立训练（不需要 LLM）
2. **Stage2**：Diffusion + LLM 联合训练

```
┌─────────────────────────────────────────────────────────┐
│              两阶段训练策略                                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Stage 1: Diffusion 独立训练                              │
│  ├─ 输入：预保存的 Hidden States                          │
│  ├─ 目标：训练 Diffusion 模型学会生成 Steps Hidden State  │
│  ├─ 不需要 LLM                                            │
│  └─ 训练时间：较快（无 LLM 前向传播）                     │
│                                                          │
│  ──────────────── 阶段切换 ────────────────               │
│                                                          │
│  Stage 2: 联合训练                                       │
│  ├─ 输入：原始数据（question, steps, answer）           │
│  ├─ 目标：端到端优化（Diffusion + LLM）                   │
│  ├─ 需要 LLM（冻结或 LoRA）                               │
│  └─ 训练时间：较慢（需要 LLM 前向传播）                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 二、Stage1：Diffusion 独立训练

### 2.1 训练目标

**目标**：让 Diffusion 模型学会从 Question Hidden State 生成 Steps Hidden State

**特点**：
- **完全解耦**：不需要 LLM 参与
- **训练快速**：无 LLM 前向传播，训练速度快
- **显存节省**：不需要加载 LLM，显存占用小

### 2.2 数据流

```
预保存的 Hidden States
    ↓
HiddenStateDataset
    ↓
DataLoader (batch_size=64)
    ↓
LitDiffLaRHidden.forward()
    ├─> question_hidden: [B, L_q, H]  # Query Last Hidden State（条件输入）
    ├─> steps_hidden: [B, L_s, H]     # Steps Hidden State (GT)（学习目标）
    └─> steps_mask: [B, L_s]
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage1a：基础去噪训练（前 50% epochs）                    │
├─────────────────────────────────────────────────────────┤
│ ├─> LatentDiffusion.forward()                          │
│ │   ├─> condition: question_hidden                      │
│ │   ├─> steps_embeds: steps_hidden                     │
│ │   ├─> 时间步采样：t ~ U[0, 1]（均匀分布）             │
│ │   └─> 输出：Diffusion Loss                            │
│ │                                                       │
│ └─> LatentDiffusion.generate()                         │
│     ├─> condition: question_hidden                      │
│     ├─> num_inference_steps: 10（快速训练）            │
│     └─> 输出：Generated Steps Hidden State              │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage1b：全噪声生成训练（后 50% epochs）                │
├─────────────────────────────────────────────────────────┤
│ ├─> LatentDiffusion.forward()                          │
│ │   ├─> condition: question_hidden                      │
│ │   ├─> steps_embeds: steps_hidden                     │
│ │   ├─> 时间步采样：t ~ U[0.8, 1.0]（强化全噪声）      │
│ │   └─> 输出：Diffusion Loss                            │
│ │                                                       │
│ └─> LatentDiffusion.generate()                         │
│     ├─> condition: question_hidden                      │
│     ├─> num_inference_steps: 20（接近推理）            │
│     └─> 输出：Generated Steps Hidden State              │
└─────────────────────────────────────────────────────────┘
    ↓
Alignment Loss（显式对齐生成和 GT）
    ↓
Total Loss = Diffusion Loss + Alignment Loss
    ↓
反向传播（只更新 Diffusion 参数）
```

### 2.3 核心代码实现

```python
def _forward_stage1(self, batch):
    """
    Stage1 前向传播：只训练 Diffusion 模型
    
    Args:
        batch: 包含预保存的 Hidden States（关键：Question 和 Steps 都需要预保存）
            - question_hidden: [B, L_q, H]  # Question Hidden State（作为条件）
            - steps_hidden: [B, L_s, H]     # Steps Hidden State（作为目标）
            - question_mask: [B, L_q]
            - steps_mask: [B, L_s]
    """
    # 1. 从 batch 中获取预保存的 Hidden States
    # 注意：Question Hidden State 也需要预保存，因为 Stage1 不需要 LLM，无法实时计算
    question_hidden = batch["question_hidden"]  # [B, L_q, H] - 预保存的
    steps_hidden = batch["steps_hidden"]  # [B, L_s, H] - 预保存的
    question_mask = batch["question_mask"].float()  # [B, L_q]
    steps_mask = batch["steps_mask"].float()  # [B, L_s]
    
    # 2. 将变长的 Steps Hidden State 转换为固定长度
    # 如果 steps_hidden 长度 < max_latent_length，需要 padding
    # 如果 steps_hidden 长度 > max_latent_length，需要 truncate
    steps_hidden_padded, steps_mask_padded = self._pad_to_fixed_length(
        steps_hidden, steps_mask, self.max_latent_length
    )
    
    # 3. 判断当前是 Stage1a 还是 Stage1b
    stage1a_epochs = int(self.stage1_epochs * self.stage1a_ratio)
    is_stage1b = self.current_epoch >= stage1a_epochs
    
    # 4. 计算 Diffusion Loss（学习去噪能力）
    # ⚠️ 关键：明确区分条件输入和学习目标
    # - condition: Question Hidden State（Query Last Hidden State）- 条件输入
    # - steps_embeds: Steps Hidden State (GT) - 学习目标
    # 
    # Stage1a：正常时间步采样（t ~ U[0, 1]）
    # Stage1b：强化全噪声采样（50% t ~ U[0.8, 1.0]，50% t ~ U[0, 1]）
    diffusion_loss, _, _ = self.latent_diffusion(
        steps_embeds=steps_hidden_padded,  # ⚠️ 学习目标：GT Steps Hidden State
        condition=question_hidden,          # ⚠️ 条件输入：Query Last Hidden State
        attention_mask=steps_mask_padded,
        condition_mask=question_mask,
        force_high_noise=is_stage1b,  # Stage1b 时强制高噪声采样
        high_noise_ratio=self.stage1b_high_noise_ratio if is_stage1b else 0.0,
    )
    
    # 5. 生成 Steps Hidden State（用于 Alignment Loss）
    # ⚠️ 关键：Diffusion 模型基于 Question Hidden State（条件）生成 Steps Hidden State
    # - 条件输入：Question Hidden State（Query Last Hidden State）
    # - 生成目标：Steps Hidden State
    # 
    # Stage1a：10 步生成（快速训练）
    # Stage1b：20 步生成（更接近推理的 32 步）
    train_inference_steps = self.stage1b_train_inference_steps if is_stage1b else self.stage1a_train_inference_steps
    generated_steps_hidden = self.latent_diffusion.generate(
        condition=question_hidden,  # ⚠️ 条件输入：Query Last Hidden State
        num_inference_steps=train_inference_steps,  # Stage1a: 10步, Stage1b: 20步
        latent_length=self.max_latent_length,
        condition_mask=question_mask,
        enable_grad=True,  # 保留梯度
        use_self_cond=True,
    )  # [B, L_s, H] - 生成的 Steps Hidden State
    
    # 5. Alignment Loss（显式对齐生成的 Hidden State 和 GT）
    # 作用：确保生成的 Steps Hidden State 接近 GT，为 Stage2 做准备
    alignment_loss = self._compute_alignment_loss(
        generated_steps_hidden, 
        steps_hidden_padded, 
        steps_mask_padded
    )
    
    # 6. Stage1 总损失：Diffusion Loss + Alignment Loss
    total_loss = (
        self.diffusion_loss_weight * diffusion_loss
        + self.alignment_loss_weight * alignment_loss
    )
    
    return {
        "total_loss": total_loss,
        "diffusion_loss": diffusion_loss,
        "alignment_loss": alignment_loss,
        "training_stage": 1.0,
        "stage1_substage": 2.0 if is_stage1b else 1.0,  # 1.0=Stage1a, 2.0=Stage1b
    }
```

### 2.4 关键实现细节

#### 2.4.1 固定长度处理

**问题**：不同样本的 Steps Hidden State 长度不同

**解决方案**：
```python
def _pad_to_fixed_length(self, hidden_states, mask, max_length):
    """
    将变长的 Hidden States 转换为固定长度
    
    Args:
        hidden_states: [B, L, H]
        mask: [B, L]
        max_length: 目标长度
    """
    batch_size, current_length, hidden_size = hidden_states.shape
    
    if current_length > max_length:
        # 截断
        hidden_states = hidden_states[:, :max_length, :]
        mask = mask[:, :max_length]
    elif current_length < max_length:
        # Padding：使用零向量（Hidden State 空间中的"无意义"表示）
        pad_length = max_length - current_length
        pad_hidden = torch.zeros(
            batch_size, pad_length, hidden_size,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        hidden_states = torch.cat([hidden_states, pad_hidden], dim=1)
        
        pad_mask = torch.zeros(
            batch_size, pad_length,
            device=mask.device, dtype=mask.dtype
        )
        mask = torch.cat([mask, pad_mask], dim=1)
    
    return hidden_states, mask
```

#### 2.4.2 归一化参数估计

**问题**：Hidden State 的分布与 Embedding 不同，需要重新估计归一化参数

**解决方案**：
```python
def _estimate_hidden_state_stats(self, datamodule):
    """从训练数据估计 Hidden State 的归一化参数"""
    train_loader = datamodule.train_dataloader()
    
    all_hidden = []
    all_masks = []
    num_samples = 0
    max_samples = 500
    
    with torch.no_grad():
        for batch in train_loader:
            steps_hidden = batch["steps_hidden"]  # [B, L, H]
            steps_mask = batch["steps_mask"]  # [B, L]
            
            all_hidden.append(steps_hidden.cpu())
            all_masks.append(steps_mask.cpu())
            num_samples += len(steps_hidden)
            
            if num_samples >= max_samples:
                break
    
    all_hidden = torch.cat(all_hidden, dim=0).to(self.device)
    all_masks = torch.cat(all_masks, dim=0).to(self.device)
    
    # 估计统计量
    mean, std, scale = self.latent_diffusion.estimate_latent_stats(
        all_hidden, all_masks
    )
    
    print(f"Hidden state stats: Mean norm={mean.norm():.4f}, "
          f"Std mean={std.mean():.6f}, Scale={scale:.4f}")
```

### 2.5 训练配置

```yaml
# Stage1 训练配置
stage1_epochs: 10              # Stage1 总 epoch 数
stage1a_ratio: 0.5             # Stage1a 占比（前 50%）
stage1b_ratio: 0.5             # Stage1b 占比（后 50%）

# Stage1a 配置（基础去噪训练）
stage1a_diffusion_loss_weight: 1.0
stage1a_alignment_loss_weight: 1.0
stage1a_train_inference_steps: 10      # 快速训练

# Stage1b 配置（全噪声生成训练）
stage1b_diffusion_loss_weight: 1.0
stage1b_alignment_loss_weight: 1.0
stage1b_train_inference_steps: 20      # 接近推理（32步）
stage1b_high_noise_ratio: 0.5          # 高噪声采样比例（t ~ U[0.8, 1.0]）

# 通用配置
diffusion_loss_weight: 1.0     # Diffusion Loss 权重
alignment_loss_weight: 1.0     # Alignment Loss 权重
```

**关键配置说明**：
- **Stage1a**：基础训练，快速学习去噪能力
  - `train_inference_steps: 10`：快速生成，节省计算
  - 时间步采样：t ~ U[0, 1]（均匀分布）
  
- **Stage1b**：全噪声强化训练，匹配推理场景
  - `train_inference_steps: 20`：更接近推理的 32 步
  - 时间步采样：50% t ~ U[0.8, 1.0]（高噪声），50% t ~ U[0, 1]（正常）
  
- **推理时**：`num_inference_steps: 32`（高质量生成）

---

## 三、Stage2：联合训练

### 3.1 训练目标

**目标**：端到端优化 Diffusion + LLM，确保生成的 Steps Hidden State 能够帮助 LLM 生成正确答案

**特点**：
- **联合优化**：同时优化 Diffusion 和 LLM（LoRA）
- **端到端**：从 Question 到 Answer 的完整流程
- **Rollout 训练**：使用生成的 Hidden State 计算 Answer Loss

### 3.2 数据流

```
原始数据 (question, steps, answer)
    ↓
Question → Tokenizer → Embedding
    ↓
LLM Forward (output_hidden_states=True)
    ↓
Question Hidden State [B, L_q, H]
    ↓
Latent Diffusion (已训练的 Stage1)
    ↓
Generated Steps Hidden State [B, L_s, H]
    ↓
├─> Alignment Loss (与 GT Steps Hidden State 对齐)
└─> Question Hidden State + Steps Hidden State → LLM Forward → Answer
    ↓
Answer Loss
    ↓
Total Loss (Diffusion + Alignment + Answer)
```

### 3.3 核心代码实现

```python
def _forward_stage2(self, batch):
    """
    Stage2 前向传播：联合训练 Diffusion + LLM
    
    Args:
        batch: 包含原始数据
            - question: List[str]
            - steps: List[str]
            - answer: List[str]
    """
    # 1. 准备输入
    question = batch["question"]
    steps = batch["steps"]
    answer = batch["answer"]
    batch_size = len(question)
    
    # 2. Question → LLM → Hidden State
    question_input_ids, question_attention_mask = self.prepare_inputs(
        question,
        padding_side="left",
        part="question",
        suffix=self.speed_template.format("auto") + self.thinking_separator,
    )
    question_embeds = self.embedding(question_input_ids)
    question_mask = question_attention_mask.float()
    
    # 获取 Question 的 Last Hidden State
    question_outputs = self.llm.forward(
        inputs_embeds=question_embeds,
        attention_mask=question_attention_mask,
        output_hidden_states=True,
    )
    question_hidden = question_outputs.hidden_states[-1]  # [B, L_q, H]
    
    # 3. 准备 GT Steps Hidden State（用于 Alignment Loss）
    steps_input_ids, steps_attention_mask = self.prepare_inputs(
        steps,
        padding_side="right",
        part="steps",
    )
    steps_embeds = self.embedding(steps_input_ids)
    steps_mask = steps_attention_mask.float()
    
    steps_outputs = self.llm.forward(
        inputs_embeds=steps_embeds,
        attention_mask=steps_attention_mask,
        output_hidden_states=True,
    )
    gt_steps_hidden = steps_outputs.hidden_states[-1]  # [B, L_s, H]
    
    # 4. Diffusion 生成 Steps Hidden State
    # 将 question_hidden 转换为固定长度（如果需要）
    question_hidden_padded, question_mask_padded = self._pad_to_fixed_length(
        question_hidden, question_mask, self.max_condition_length
    )
    
    generated_steps_hidden = self.latent_diffusion.generate(
        condition=question_hidden_padded,
        num_inference_steps=self.train_inference_steps,
        latent_length=self.max_latent_length,
        condition_mask=question_mask_padded,
        enable_grad=True,  # 保留梯度
        use_self_cond=True,
    )  # [B, L_s, H]
    
    # 5. 将变长的 GT Steps Hidden State 转换为固定长度
    gt_steps_hidden_padded, steps_mask_padded = self._pad_to_fixed_length(
        gt_steps_hidden, steps_mask, self.max_latent_length
    )
    
    # 6. Diffusion Loss（仍用 GT 作为目标）
    # ⚠️ 关键：明确区分条件输入和学习目标
    # - condition: Question Hidden State（Query Last Hidden State）- 条件输入
    # - steps_embeds: Steps Hidden State (GT) - 学习目标
    diffusion_loss, _, _ = self.latent_diffusion(
        steps_embeds=gt_steps_hidden_padded,  # ⚠️ 学习目标：GT Steps Hidden State
        condition=question_hidden_padded,      # ⚠️ 条件输入：Query Last Hidden State
        attention_mask=steps_mask_padded,
        condition_mask=question_mask_padded,
        use_self_cond=True,
    )
    
    # 7. Alignment Loss（对齐生成和 GT）
    alignment_loss = self._compute_alignment_loss(
        generated_steps_hidden, gt_steps_hidden_padded, steps_mask_padded
    )
    
    # 8. Answer Loss（使用生成的 Hidden State）
    answer_input_ids, answer_attention_mask = self.prepare_inputs(
        answer,
        padding_side="right",
        part="answer",
        prefix=self.thinking_separator,
        suffix=self.tokenizer.eos_token,
    )
    answer_embeds = self.embedding(answer_input_ids)
    
    # 拼接：Question Hidden State + Generated Steps Hidden State + Answer Embedding
    # 关键：Question 和 Steps 都使用 Hidden State，保持一致性
    all_embeds = torch.cat([
        question_hidden,  # Question Hidden State
        generated_steps_hidden,  # Steps Hidden State
        answer_embeds  # Answer Embedding
    ], dim=1)
    
    # 注意：question_hidden 的长度可能与 question_attention_mask 不同（如果进行了 padding/truncate）
    question_hidden_length = question_hidden.shape[1]
    all_attention_mask = torch.cat([
        torch.ones(batch_size, question_hidden_length, device=self.device, dtype=question_attention_mask.dtype),
        torch.ones(batch_size, self.max_latent_length, device=self.device, dtype=question_attention_mask.dtype),
        answer_attention_mask
    ], dim=1)
    
    # 创建 labels（只计算 answer 部分的 loss）
    question_length = question_hidden.shape[1]
    steps_length = generated_steps_hidden.shape[1]
    
    labels = torch.cat([
        torch.full((batch_size, question_length + steps_length), -100, device=self.device),
        answer_input_ids,
    ], dim=1)
    labels[labels == self.tokenizer.pad_token_id] = -100
    
    position_ids = get_position_ids_from_attention_mask(all_attention_mask)
    
    answer_outputs = self.llm.forward(
        inputs_embeds=all_embeds,
        attention_mask=all_attention_mask,
        position_ids=position_ids,
        labels=labels,
    )
    answer_loss = answer_outputs.loss
    
    # 9. 总损失
    total_loss = (
        self.diffusion_loss_weight * diffusion_loss
        + self.alignment_loss_weight * alignment_loss
        + self.answer_loss_weight * answer_loss
    )
    
    return {
        "total_loss": total_loss,
        "diffusion_loss": diffusion_loss,
        "alignment_loss": alignment_loss,
        "answer_loss": answer_loss,
        "training_stage": 2.0,
    }
```

### 3.4 关键实现细节

#### 3.4.1 Hidden State → LLM 输入

**问题**：LLM 的 `forward` 方法通常接受 `inputs_embeds`（Embedding），但我们现在有 Hidden State

**解决方案**：
1. **方案1**：直接使用 Hidden State（如果 LLM 支持）
2. **方案2**：通过投影层将 Hidden State 映射回 Embedding 空间
3. **方案3**：使用 LLM 的中间层输入（需要修改 LLM 代码）

**推荐方案1**：直接使用 Hidden State，因为 Hidden State 和 Embedding 维度相同，LLM 应该能够接受。

#### 3.4.2 Alignment Loss

```python
def _compute_alignment_loss(
    self,
    generated_hidden: torch.Tensor,
    gt_hidden: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算 Hidden State 对齐损失
    
    Args:
        generated_hidden: [B, L, H]
        gt_hidden: [B, L, H]
        mask: [B, L]
    """
    mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
    
    # MSE Loss
    mse_loss = F.mse_loss(
        generated_hidden * mask_expanded,
        gt_hidden * mask_expanded,
        reduction='sum'
    ) / (mask.sum() * generated_hidden.shape[-1] + 1e-8)
    
    # Cosine Similarity Loss
    gen_norm = F.normalize(generated_hidden, p=2, dim=-1)
    gt_norm = F.normalize(gt_hidden, p=2, dim=-1)
    cosine_sim = (gen_norm * gt_norm).sum(dim=-1)  # [B, L]
    cosine_loss = (1 - cosine_sim) * mask
    cosine_loss = cosine_loss.sum() / (mask.sum() + 1e-8)
    
    total_alignment_loss = mse_loss + 0.1 * cosine_loss
    return total_alignment_loss
```

### 3.5 训练配置

```yaml
# Stage2 训练配置
stage2_epochs: 10              # Stage2 训练的 epoch 数
diffusion_loss_weight: 1.0     # Diffusion Loss 权重
alignment_loss_weight: 1.0      # Alignment Loss 权重
answer_loss_weight: 5.0         # Answer Loss 权重
train_inference_steps: 10       # 训练时 Diffusion 推理步数
```

---

## 四、阶段切换

### 4.1 自动切换

```python
def on_train_epoch_start(self):
    """每个 epoch 开始时检查是否切换阶段"""
    if self.current_epoch >= self.stage1_epochs and self.training_stage == 1:
        self.training_stage = 2
        
        # 加载 LLM（如果 Stage1 没有加载）
        if not hasattr(self, 'llm') or self.llm is None:
            self._load_llm()
        
        print(f"\n{'='*60}")
        print(f"Switching to Stage 2: Joint Training")
        print(f"{'='*60}\n")
```

### 4.2 检查点管理

**Stage1 检查点**：只保存 Diffusion 模型参数
**Stage2 检查点**：保存 Diffusion + LLM LoRA 参数

---

## 五、训练监控

### 5.1 Loss 记录

```python
def training_step(self, batch, batch_idx):
    log_dict = self.forward(batch=batch)
    
    # 记录 Loss
    self.log_dict({
        "train/total_loss": log_dict["total_loss"],
        "train/diffusion_loss": log_dict["diffusion_loss"],
        "train/alignment_loss": log_dict.get("alignment_loss", 0.0),
        "train/answer_loss": log_dict.get("answer_loss", 0.0),
        "train/stage": float(self.training_stage),
    }, sync_dist=True, prog_bar=True)
    
    return log_dict["total_loss"]
```

### 5.2 可视化

- Loss 曲线（Total, Diffusion, Alignment, Answer）
- 阶段切换点标记
- 训练进度（Stage1/Stage2）

---

*文档版本：v1.0*  
*创建时间：2026-01-12*


