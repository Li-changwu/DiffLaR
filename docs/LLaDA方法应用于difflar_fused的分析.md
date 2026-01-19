# 使用 LLaDA 方法解决 difflar_fused 中的 Steps Embedding 离散和前向去噪问题

## 一、问题分析

### 1.1 difflar_fused 的当前问题

根据代码分析，`difflar_fused` 存在以下核心问题：

1. **Steps Embedding 离散化问题**：
   - 当前方法：在连续的 embedding 空间进行扩散，生成连续的 latent
   - 问题：生成的 embedding 需要映射回离散的 token，存在量化误差
   - 位置：`prepare_fixed_length_steps()` 将文本转换为 embedding，但反向过程（embedding → token）不明确

2. **前向去噪过程较长**：
   - 当前配置：推理时需要 128 步去噪（`num_inference_steps=128`）
   - 训练时：仅 10 步（`train_inference_steps=10`），训练-推理分布差异大
   - 问题：去噪步数多导致推理慢，且精度可能不足

### 1.2 LLaDA 的核心优势

LLaDA 使用 **Masked Diffusion Model (MDM)**，具有以下特点：

1. **直接在 Token 空间操作**：
   - 使用离散的 `[MASK]` token（ID=126336）而非连续噪声
   - 避免了 embedding 空间的离散化问题
   - 不需要时间步 t 作为 Transformer 输入（RADD 理论贡献）

2. **Remasking 策略**：
   - 在去噪过程中重新 mask 低置信度的 token
   - 可以加速去噪过程，减少所需步数

3. **Block Diffusion**：
   - 使用半自回归的 block 生成
   - 可以并行生成多个 block，进一步加速

---

## 二、LLaDA 方法的核心机制

### 2.1 Masked Diffusion 训练过程

```python
# LLaDA 的训练代码（来自 GUIDELINES.md）
def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)  # 随机采样时间步
    p_mask = (1 - eps) * t + eps  # mask 概率
    p_mask = p_mask[:, None].repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 是 [MASK] token ID
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

# 训练损失
noisy_batch, masked_indices, p_mask = forward_process(input_ids)
logits = model(input_ids=noisy_batch).logits
token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
```

**关键点**：
- 直接在 token 空间操作，不需要 embedding
- 使用离散的 `[MASK]` token 替代连续噪声
- 损失函数包含 `1/p_mask` 权重，这是 masked diffusion 的核心

### 2.2 Remasking 去噪过程

```python
# LLaDA 的生成代码（来自 generate.py）
def generate(model, prompt, steps=128, gen_length=128, remasking='low_confidence'):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    for i in range(steps):
        mask_index = (x == mask_id)
        logits = model(x).logits
        
        # 预测所有位置的 token
        x0 = torch.argmax(logits, dim=-1)
        
        # Remasking：计算置信度
        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == 'random':
            x0_p = torch.rand_like(x0, dtype=torch.float)
        
        # 只更新高置信度的位置
        confidence = torch.where(mask_index, x0_p, -np.inf)
        num_transfer = mask_num // steps  # 每步更新的 token 数
        _, select_index = torch.topk(confidence, k=num_transfer)
        x[select_index] = x0[select_index]
        
        # 低置信度的位置重新 mask
        # （通过不更新这些位置实现）
    
    return x
```

**关键点**：
- 每步只更新部分高置信度的 token
- 低置信度的 token 保持 mask 状态，等待后续步骤
- 可以显著减少所需步数

---

## 三、将 LLaDA 方法应用到 difflar_fused

### 3.1 方案 1：完全替换为 Masked Diffusion（推荐）

**核心思想**：将 `difflar_fused` 的连续扩散替换为 LLaDA 的 masked diffusion。

#### 3.1.1 修改训练过程

```python
# 修改 prepare_fixed_length_steps，返回 token IDs 而非 embedding
def prepare_fixed_length_steps_tokens(
    self,
    steps: List[str],
    max_length: Optional[int] = None,
) -> tuple:
    """将变长的Steps文本转换为固定长度的Token IDs"""
    if max_length is None:
        max_length = self.max_latent_length
    
    steps_text = [self.steps_template.format(s) for s in steps]
    inputs = self.tokenizer.batch_encode_plus(
        steps_text,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    steps_input_ids = inputs["input_ids"].to(self.device)
    steps_attention_mask = inputs["attention_mask"].to(self.device)
    
    return steps_input_ids, steps_attention_mask

# 添加 Masked Diffusion 训练函数
def forward_masked_diffusion(
    self,
    steps_input_ids: torch.Tensor,
    condition: torch.Tensor,  # query embedding
    mask: torch.Tensor,
    eps: float = 1e-3,
):
    """
    Masked Diffusion 训练过程（类似 LLaDA）
    
    Args:
        steps_input_ids: [B, L] GT token IDs
        condition: [B, L_q, H] query embedding
        mask: [B, L] 有效位置掩码
        eps: 最小 mask 概率
    """
    batch_size, seq_length = steps_input_ids.shape
    
    # 1. 随机采样时间步 t ~ U[0, 1]
    t = torch.rand(batch_size, device=self.device)
    
    # 2. 计算 mask 概率 p_mask = (1-eps)*t + eps
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].expand(batch_size, seq_length)
    
    # 3. 随机 mask（只在有效位置）
    random_mask = torch.rand_like(p_mask) < p_mask
    masked_indices = random_mask & (mask.bool())
    
    # 4. 创建 noisy batch（mask 位置替换为 [MASK] token）
    mask_token_id = 126336  # 或使用 tokenizer.mask_token_id
    noisy_input_ids = torch.where(
        masked_indices,
        torch.full_like(steps_input_ids, mask_token_id),
        steps_input_ids
    )
    
    # 5. 将 condition embedding 和 noisy tokens 结合
    # 方法：将 condition 作为 prefix，noisy tokens 作为后续
    condition_embeds = condition  # [B, L_q, H]
    noisy_embeds = self.embedding(noisy_input_ids)  # [B, L, H]
    
    # 6. 拼接并输入 LLM（移除 causal mask，使用 encoder 模式）
    all_embeds = torch.cat([condition_embeds, noisy_embeds], dim=1)
    all_mask = torch.cat([
        torch.ones(batch_size, condition.shape[1], device=self.device),
        mask
    ], dim=1)
    
    # 7. 获取 logits（需要修改 LLM 为 encoder 模式）
    logits = self.llm(inputs_embeds=all_embeds, attention_mask=all_mask).logits
    
    # 8. 只取 steps 部分的 logits
    steps_logits = logits[:, condition.shape[1]:, :]
    
    # 9. 计算损失（只在 masked 位置）
    token_loss = F.cross_entropy(
        steps_logits[masked_indices],
        steps_input_ids[masked_indices],
        reduction='none'
    ) / p_mask[masked_indices]
    
    # 10. 平均损失（按有效位置）
    loss = token_loss.sum() / (mask.sum() + 1e-8)
    
    return loss
```

#### 3.1.2 修改生成过程

```python
@torch.no_grad()
def generate_masked_diffusion(
    self,
    query_embedding: torch.Tensor,
    query_mask: torch.Tensor,
    gen_length: int = 256,
    steps: int = 50,  # 可以大幅减少步数
    remasking: str = 'low_confidence',
):
    """
    Masked Diffusion 生成过程（类似 LLaDA）
    
    Args:
        query_embedding: [B, L_q, H] query embedding
        query_mask: [B, L_q] query mask
        gen_length: 生成长度
        steps: 去噪步数（可以大幅减少，如 20-50 步）
        remasking: 'low_confidence' 或 'random'
    """
    batch_size = query_embedding.shape[0]
    mask_token_id = 126336
    
    # 1. 初始化：全 mask
    steps_input_ids = torch.full(
        (batch_size, gen_length),
        mask_token_id,
        dtype=torch.long,
        device=self.device
    )
    
    # 2. 拼接 condition 和 steps
    condition_embeds = query_embedding
    steps_embeds = self.embedding(steps_input_ids)
    all_embeds = torch.cat([condition_embeds, steps_embeds], dim=1)
    all_mask = torch.cat([
        query_mask,
        torch.ones(batch_size, gen_length, device=self.device)
    ], dim=1)
    
    # 3. 计算每步需要更新的 token 数
    mask_index = (steps_input_ids == mask_token_id)
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = mask_num // steps
    
    # 4. 迭代去噪
    for i in range(steps):
        # 4.1 获取 logits
        logits = self.llm(inputs_embeds=all_embeds, attention_mask=all_mask).logits
        steps_logits = logits[:, condition_embeds.shape[1]:, :]
        
        # 4.2 预测所有位置的 token
        x0 = torch.argmax(steps_logits, dim=-1)
        
        # 4.3 计算置信度（用于 remasking）
        if remasking == 'low_confidence':
            p = F.softmax(steps_logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == 'random':
            x0_p = torch.rand_like(x0, dtype=torch.float)
        
        # 4.4 只更新高置信度的位置
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        
        # 4.5 选择 top-k 高置信度位置更新
        for j in range(batch_size):
            _, select_index = torch.topk(
                confidence[j],
                k=num_transfer_tokens[j, 0].item()
            )
            steps_input_ids[j, select_index] = x0[j, select_index]
        
        # 4.6 更新 mask_index 和 embedding
        mask_index = (steps_input_ids == mask_token_id)
        steps_embeds = self.embedding(steps_input_ids)
        all_embeds = torch.cat([condition_embeds, steps_embeds], dim=1)
    
    return steps_input_ids
```

#### 3.1.3 优势

1. **解决离散化问题**：
   - 直接在 token 空间操作，无需 embedding → token 映射
   - 避免了量化误差

2. **加速去噪**：
   - 可以使用更少的步数（20-50 步 vs 128 步）
   - Remasking 策略可以智能地选择更新哪些位置

3. **训练-推理一致性**：
   - 训练和推理都在 token 空间，分布一致
   - 减少 Exposure Bias

### 3.2 方案 2： hybrid 方法（保留 embedding 空间，但使用 mask 策略）

**核心思想**：保留 embedding 空间的扩散，但引入 LLaDA 的 remasking 策略。

#### 3.2.1 修改生成过程

```python
@torch.no_grad()
def generate_with_remasking(
    self,
    query_embedding: torch.Tensor,
    query_mask: torch.Tensor,
    gen_length: int = 256,
    steps: int = 50,  # 减少步数
    remasking: str = 'low_confidence',
):
    """
    使用 Remasking 策略的生成过程
    
    核心改进：
    1. 每步只更新高置信度的位置
    2. 低置信度位置保持噪声状态，等待后续步骤
    """
    batch_size = query_embedding.shape[0]
    
    # 1. 初始化：纯噪声
    steps_embeds = torch.randn(
        batch_size, gen_length, self.hidden_size,
        device=self.device
    )
    
    # 2. 归一化
    if self.latent_diffusion.normalize_latent:
        steps_embeds = self.latent_diffusion.normalize(steps_embeds)
    
    # 3. 创建置信度 mask（初始全为 True，表示所有位置都需要更新）
    confidence_mask = torch.ones(
        batch_size, gen_length,
        dtype=torch.bool,
        device=self.device
    )
    
    # 4. 迭代去噪
    for i in range(steps):
        # 4.1 预测速度/噪声
        t = 1.0 - (i + 1) / steps  # 从 1.0 到 0.0
        t_discrete = int(t * self.latent_diffusion.num_timesteps)
        
        predicted = self.latent_diffusion.denoiser(
            steps_embeds,
            torch.full((batch_size,), t_discrete, device=self.device),
            query_embedding,
            attention_mask=confidence_mask.float(),
            condition_mask=query_mask,
        )
        
        # 4.2 计算置信度（基于预测的方差或熵）
        if remasking == 'low_confidence':
            # 方法1：基于预测的方差
            # 预测的方差越小，置信度越高
            variance = predicted.var(dim=-1)  # [B, L]
            confidence = 1.0 / (1.0 + variance)  # 归一化到 [0, 1]
        elif remasking == 'random':
            confidence = torch.rand(batch_size, gen_length, device=self.device)
        
        # 4.3 选择高置信度位置更新
        num_update = gen_length // steps  # 每步更新的位置数
        _, top_indices = torch.topk(confidence, k=num_update, dim=1)
        
        # 4.4 只更新高置信度位置
        update_mask = torch.zeros_like(confidence_mask)
        for j in range(batch_size):
            update_mask[j, top_indices[j]] = True
        
        # 4.5 欧拉积分更新（只更新高置信度位置）
        dt = 1.0 / steps
        update_embeds = steps_embeds - dt * predicted
        steps_embeds = torch.where(
            update_mask.unsqueeze(-1),
            update_embeds,
            steps_embeds  # 低置信度位置保持原状
        )
        
        # 4.6 更新置信度 mask（已更新的位置不再需要更新）
        confidence_mask = confidence_mask & (~update_mask)
    
    # 5. 反归一化
    if self.latent_diffusion.normalize_latent:
        steps_embeds = self.latent_diffusion.denormalize(steps_embeds)
    
    return steps_embeds
```

#### 3.2.2 优势

1. **保留现有架构**：
   - 不需要大幅修改代码
   - 保留 embedding 空间的优势

2. **加速去噪**：
   - Remasking 策略可以减少所需步数
   - 智能选择更新位置

---

## 四、实施建议

### 4.1 推荐方案

**优先尝试方案 1（完全替换为 Masked Diffusion）**，原因：

1. **彻底解决离散化问题**：直接在 token 空间操作
2. **理论支持**：LLaDA 已经证明了 masked diffusion 的有效性
3. **加速效果明显**：可以大幅减少推理步数

### 4.2 实施步骤

1. **第一步：修改训练过程**
   - 实现 `forward_masked_diffusion()` 函数
   - 修改 `prepare_fixed_length_steps()` 返回 token IDs
   - 修改 LLM 为 encoder 模式（移除 causal mask）

2. **第二步：修改生成过程**
   - 实现 `generate_masked_diffusion()` 函数
   - 添加 remasking 策略
   - 支持 block diffusion（可选）

3. **第三步：调整超参数**
   - `steps`: 从 128 减少到 20-50
   - `remasking`: 使用 'low_confidence'
   - `eps`: 设置为 1e-3（LLaDA 默认值）

4. **第四步：实验验证**
   - 对比精度和速度
   - 调整超参数

### 4.3 注意事项

1. **LLM 架构修改**：
   - 需要将 LLM 从 decoder 模式改为 encoder 模式
   - 移除 causal mask，允许双向注意力

2. **Mask Token**：
   - 需要确保 tokenizer 有 `[MASK]` token
   - 或使用特殊 token ID（如 126336）

3. **训练稳定性**：
   - Masked diffusion 的训练可能比连续扩散更稳定
   - 但仍需要监控 loss 曲线

---

## 五、预期效果

### 5.1 精度提升

- **解决离散化误差**：直接在 token 空间操作，避免 embedding → token 映射误差
- **训练-推理一致性**：训练和推理都在 token 空间，减少 Exposure Bias

### 5.2 速度提升

- **减少推理步数**：从 128 步减少到 20-50 步（约 2.5-6 倍加速）
- **Remasking 策略**：智能选择更新位置，进一步提高效率

### 5.3 训练稳定性

- **更简单的训练目标**：token 预测比连续扩散更直观
- **更好的收敛性**：LLaDA 已经证明了 masked diffusion 的有效性

---

## 六、总结

LLaDA 的 **Masked Diffusion Model** 方法可以有效解决 `difflar_fused` 中的两个核心问题：

1. **Steps Embedding 离散化问题**：
   - 通过直接在 token 空间操作，完全避免了离散化误差

2. **前向去噪过程较长**：
   - 通过 remasking 策略和减少步数，可以大幅加速推理

**推荐实施方案**：完全替换为 Masked Diffusion（方案 1），这是最彻底的解决方案，且已有 LLaDA 的成功验证。

---

*分析时间：2026-01-09*
*基于代码版本：最新版本*





