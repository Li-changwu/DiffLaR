# DiffLaR Hidden 算法概述

## 一、核心思想

### 1.1 问题背景

在之前的实验中观察到，**Embedding 空间仍然是离散的高维向量空间**，对于 Diffusion 模型这种在连续空间中探索的模型来说，学习离散的高维向量能力确实不太强。

### 1.2 核心创新

**关键洞察**：大模型最后一个隐藏层的输出（Last Hidden State）是**连续的**，比 Embedding 更适合 Diffusion 模型学习。

**核心改进**：
- **从 Embedding 空间 → Last Hidden State 空间**
- **Stage1 完全解耦**：不需要大模型参与，直接使用预保存的 Last Hidden States 训练 Diffusion
- **Stage2 联合训练**：Diffusion 训练完成后，再接入 LLM 进行联合训练

### 1.3 与 DiffLaR Fused 的对比

| 维度 | DiffLaR Fused | DiffLaR Hidden |
|------|---------------|----------------|
| **学习目标** | Steps 对应的 Embedding | Steps 对应的 Last Hidden State |
| **空间特性** | 离散的高维向量空间 | 连续的特征空间 |
| **Stage1 训练** | 需要 LLM 参与（冻结） | **完全不需要 LLM** |
| **数据准备** | 实时计算 Embedding | **预提取并保存 Last Hidden States** |
| **训练效率** | 需要前向传播获取 Embedding | 直接加载预保存数据，更快 |
| **推理步数** | 128 步 | **32 步**（降低计算成本） |

---

## 二、算法优势

### 2.1 理论优势

1. **连续空间更适合 Diffusion**
   - Last Hidden State 是 LLM 内部连续表示
   - 比离散的 Embedding 空间更平滑，梯度更稳定
   - Diffusion 模型在连续空间中表现更好

2. **训练解耦**
   - Stage1 完全独立，可以快速迭代 Diffusion 模型
   - 不需要加载大模型，节省显存和计算资源
   - 可以并行处理多个实验

3. **数据复用**
   - Last Hidden States 可以预计算并保存
   - 一次提取，多次使用
   - 支持大规模数据集的快速训练

### 2.2 实践优势

1. **训练速度**
   - Stage1 不需要 LLM 前向传播，训练更快
   - 数据加载更快（直接加载 tensor）

2. **显存占用**
   - Stage1 不需要加载 LLM，显存占用更小
   - 可以支持更大的 batch size

3. **推理效率**
   - 推理步数从 128 步减少到 32 步
   - 推理速度提升约 4 倍

---

## 三、算法流程概览

```
┌─────────────────────────────────────────────────────────────┐
│              DiffLaR Hidden 完整流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  【数据预处理阶段】（一次性）                                  │
│  ┌────────────────────────────────────────────┐             │
│  │ 1. 加载训练数据（question, steps, answer）  │             │
│  │ 2. 对每个样本：                              │             │
│  │    - Question → LLM forward                │             │
│  │    - Steps → LLM forward                   │             │
│  │    - 提取 Last Hidden State                │             │
│  │ 3. 保存到本地（.pt 文件）                   │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  【Stage1：Diffusion 独立训练】（不需要 LLM）                │
│  ┌────────────────────────────────────────────┐             │
│  │ 1. 加载预保存的 Last Hidden States          │             │
│  │    - Question Hidden State（条件输入）     │             │
│  │      ⚠️ 作为 Diffusion 的条件输入           │             │
│  │    - Steps Hidden State（学习目标）         │             │
│  │      ⚠️ 作为 Diffusion 的学习目标           │             │
│  │ 2. Stage1a：基础去噪训练（前 50%）          │             │
│  │    - Diffusion Loss：正常时间步采样         │             │
│  │    - Alignment Loss：10 步生成             │             │
│  │ 3. Stage1b：全噪声生成训练（后 50%）         │             │
│  │    - Diffusion Loss：强化高噪声采样         │             │
│  │      (t ~ U[0.8, 1.0])                     │             │
│  │    - Alignment Loss：20 步生成             │             │
│  │      (更接近推理的 32 步)                    │             │
│  │ 4. 只训练 Diffusion 模型参数                │             │
│  │    ⚠️ 注意：Question 和 Steps 都需要预保存  │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  【Stage2：联合训练】（需要 LLM）                             │
│  ┌────────────────────────────────────────────┐             │
│  │ 1. 加载训练好的 Diffusion 模型              │             │
│  │ 2. 加载 LLM（冻结或 LoRA）                  │             │
│  │ 3. 联合训练：                                │             │
│  │    - Diffusion 生成 Steps Hidden State     │             │
│  │    - Question Hidden State + Steps Hidden   │             │
│  │      State → LLM 生成 Answer                │             │
│  │ 4. 端到端优化                                │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  【推理阶段】                                                 │
│  ┌────────────────────────────────────────────┐             │
│  │ 1. Question → LLM → Question Hidden State  │             │
│  │ 2. Diffusion 生成 Steps Hidden State (32步) │             │
│  │ 3. Question Hidden State + Steps Hidden    │             │
│  │    State → LLM → Answer                    │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、关键技术点

### 4.1 Last Hidden State 的提取

**位置**：LLM 的最后一层 Transformer 输出

**获取方式**：
```python
outputs = llm.forward(
    inputs_embeds=steps_embeds,
    output_hidden_states=True
)
last_hidden_state = outputs.hidden_states[-1]  # [B, L, H]
```

**特点**：
- 维度与 Embedding 相同：`[batch_size, seq_length, hidden_size]`
- 包含更丰富的语义信息（经过多层 Transformer 处理）
- 是连续的特征表示

### 4.2 数据存储格式

**文件结构**：
```
data/
├── hidden_states/
│   ├── train/
│   │   ├── question_hidden_states.pt      # [N, L_q, H]
│   │   ├── steps_hidden_states.pt         # [N, L_s, H]
│   │   ├── question_attention_mask.pt     # [N, L_q]
│   │   ├── steps_attention_mask.pt        # [N, L_s]
│   │   └── metadata.json                  # 数据元信息
│   └── val/
│       └── ...
```

**优势**：
- 一次提取，多次使用
- 支持快速加载（直接 torch.load）
- 可以并行处理多个实验

### 4.3 推理步数优化

**从 128 步 → 32 步**：
- **原因**：Last Hidden State 空间更连续，需要更少的去噪步数
- **方法**：使用 Flow Matching + 更少的推理步数
- **效果**：推理速度提升约 4 倍，质量损失可接受

---

## 五、预期效果

### 5.1 训练效率

- **Stage1 训练速度**：提升 3-5 倍（不需要 LLM 前向传播）
- **显存占用**：减少 40-60%（Stage1 不需要 LLM）
- **数据加载速度**：提升 2-3 倍（直接加载 tensor）

### 5.2 推理效率

- **推理速度**：提升约 4 倍（32 步 vs 128 步）
- **推理质量**：预期保持或略有提升（连续空间更适合 Diffusion）

### 5.3 模型质量

- **收敛速度**：预期更快（连续空间梯度更稳定）
- **最终性能**：预期与 DiffLaR Fused 相当或更好

---

## 六、文档结构

本算法设计文档分为以下几个部分：

1. **算法概述**（本文档）：核心思想、优势、流程概览
2. **架构设计**：整体架构、模块组成、数据流
3. **数据预处理**：Last Hidden State 提取、存储格式、加载方式
4. **训练流程**：Stage1 和 Stage2 的详细训练流程
5. **推理流程**：推理时的完整流程和优化
6. **配置参数**：所有配置项的详细说明

---

*文档版本：v1.0*  
*创建时间：2026-01-12*


