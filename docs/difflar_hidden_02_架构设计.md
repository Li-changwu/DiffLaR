# DiffLaR Hidden 架构设计

## 一、整体架构

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    DiffLaR Hidden 系统架构                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              数据预处理模块（一次性）                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │ Question     │  │ Steps        │  │ Answer       │   │  │
│  │  │ (文本)        │  │ (文本)        │  │ (文本)        │   │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────────┘   │  │
│  │         │                  │                             │  │
│  │         ▼                  ▼                             │  │
│  │  ┌──────────────────────────────────────┐               │  │
│  │  │         LLM Forward (一次性)          │               │  │
│  │  │  output_hidden_states=True            │               │  │
│  │  └──────┬───────────────────┬────────────┘               │  │
│  │         │                   │                             │  │
│  │         ▼                   ▼                             │  │
│  │  Question Hidden State  Steps Hidden State               │  │
│  │  [B, L_q, H]              [B, L_s, H]                    │  │
│  │         │                   │                             │  │
│  │         └─────────┬─────────┘                             │  │
│  │                   ▼                                       │  │
│  │         ┌──────────────────┐                              │  │
│  │         │  保存到本地文件   │                              │  │
│  │         │  (.pt 格式)      │                              │  │
│  │         └──────────────────┘                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Stage1：Diffusion 独立训练（不需要 LLM）           │  │
│  │                                                           │  │
│  │  ┌──────────────────┐         ┌──────────────────┐     │  │
│  │  │ 加载预保存数据     │         │ 加载预保存数据     │     │  │
│  │  │ Question Hidden  │         │ Steps Hidden     │     │  │
│  │  │ State            │         │ State (GT)       │     │  │
│  │  │ (条件输入)        │         │ (学习目标)        │     │  │
│  │  └──────┬───────────┘         └──────┬───────────┘     │  │
│  │         │                            │                  │  │
│  │         │                            │                  │  │
│  │         ▼                            ▼                  │  │
│  │         ┌──────────────────────────────────────────┐   │  │
│  │         │   Latent Diffusion                      │   │  │
│  │         │   ┌──────────────────────────────────┐  │   │  │
│  │         │   │   Denoiser                       │  │   │  │
│  │         │   │   (Transformer)                  │  │   │  │
│  │         │   │   - 条件输入: Question Hidden    │  │   │  │
│  │         │   │   - 学习目标: Steps Hidden (GT)  │  │   │  │
│  │         │   └──────────────────────────────────┘  │   │  │
│  │         └──────────┬───────────────────────────────┘   │  │
│  │                    │                                    │  │
│  │  ┌─────────────────┴─────────────────┐                 │  │
│  │  │                                   │                 │  │
│  │  ▼                                   ▼                 │  │
│  │  ┌──────────────────┐  ┌──────────────────────────┐   │  │
│  │  │ Stage1a          │  │ Stage1b                  │   │  │
│  │  │ (基础训练)        │  │ (全噪声强化)             │   │  │
│  │  │                  │  │                        │   │  │
│  │  │ t ~ U[0, 1]      │  │ t ~ U[0.8, 1.0]        │   │  │
│  │  │ 10步生成         │  │ 20步生成                │   │  │
│  │  └──────────────────┘  └──────────────────────────┘   │  │
│  │         │                            │                  │  │
│  │         ├──────────────────┐         │                  │  │
│  │         │                  │         │                  │  │
│  │         ▼                  ▼         ▼                  │  │
│  │  ┌──────────────────┐  ┌──────────────────┐             │  │
│  │  │ Diffusion Loss   │  │ Generate Steps   │             │  │
│  │  │ (学习去噪能力)    │  │ Hidden State     │             │  │
│  │  └──────────────────┘  └──────┬───────────┘             │  │
│  │                               │                        │  │
│  │                               ▼                        │  │
│  │                     ┌──────────────────┐               │  │
│  │                     │ Alignment Loss   │               │  │
│  │                     │ (显式对齐生成和GT)│               │  │
│  │                     └──────────────────┘               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Stage2：联合训练（需要 LLM）                     │  │
│  │                                                           │  │
│  │  ┌──────────────────┐                                    │  │
│  │  │ Question         │                                    │  │
│  │  │ (文本)            │                                    │  │
│  │  └──────┬───────────┘                                    │  │
│  │         │                                                │  │
│  │         ▼                                                │  │
│  │  ┌──────────────────┐                                    │  │
│  │  │ LLM Forward      │                                    │  │
│  │  │ (冻结或 LoRA)     │                                    │  │
│  │  └──────┬───────────┘                                    │  │
│  │         │                                                │  │
│  │         ▼                                                │  │
│  │  Question Hidden State [B, L_q, H]                      │  │
│  │         │                                                │  │
│  │         ▼                                                │  │
│  │  ┌──────────────────────────┐                           │  │
│  │  │  Latent Diffusion        │                           │  │
│  │  │  (已训练好的 Stage1)      │                           │  │
│  │  └──────┬───────────────────┘                           │  │
│  │         │                                                │  │
│  │         ▼                                                │  │
│  │  Generated Steps Hidden State [B, L_s, H]              │  │
│  │         │                                                │  │
│  │         ├──────────────────┐                            │  │
│  │         │                  │                            │  │
│  │         ▼                  │                            │  │
│  │  ┌──────────────┐          │                            │  │
│  │  │ Alignment    │          │                            │  │
│  │  │ Loss         │          │                            │  │
│  │  └──────────────┘          │                            │  │
│  │                            │                            │  │
│  │  Question Hidden State ────┼───┐                        │  │
│  │                            │   │                        │  │
│  │                            ▼   ▼                        │  │
│  │                    ┌──────────────────┐                 │  │
│  │                    │  Concat:        │                 │  │
│  │                    │  [Q_H, S_H]     │                 │  │
│  │                    └──────┬───────────┘                 │  │
│  │                           │                             │  │
│  │                           ▼                             │  │
│  │                  ┌──────────────┐                       │  │
│  │                  │ LLM Forward  │                       │  │
│  │                  │ (生成 Answer) │                       │  │
│  │                  └──────┬───────┘                       │  │
│  │                           │                             │  │
│  │                           ▼                             │  │
│  │                  Answer Loss                            │  │
│  │                           │                             │  │
│  │                           ▼                             │  │
│  │                  Total Loss                             │  │
│  │                  (Diffusion + Alignment + Answer)       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 模块组成

| 模块 | 类名/文件 | 功能 | 训练状态 |
|------|----------|------|---------|
| **主模型** | `LitDiffLaRHidden` | 协调各模块，实现两阶段训练 | - |
| **数据预处理** | `extract_hidden_states.py` | 提取并保存 Last Hidden States | 一次性 |
| **LLM** | `LlamaForCausalLM` | 生成答案（Stage2） | 冻结或 LoRA |
| **Latent Diffusion** | `LatentDiffusion` | 扩散模型核心 | **训练** |
| **Denoiser** | `Denoiser` | 去噪网络 | **训练** |
| **Hidden State Loader** | `HiddenStateDataset` | 加载预保存的 Hidden States | 数据加载 |

---

## 二、核心模块设计

### 2.1 LitDiffLaRHidden 主模型

**文件路径**：`src/models/difflar_hidden.py`

**核心功能**：
1. **Stage1 训练**：只训练 Diffusion 模型，使用预保存的 Hidden States
2. **Stage2 训练**：联合训练 Diffusion + LLM
3. **推理**：生成 Steps Hidden State，然后生成 Answer

**关键方法**：
```python
class LitDiffLaRHidden(LitCoTModelBase):
    def __init__(self, ...):
        # 初始化 Diffusion 模型
        self.latent_diffusion = LatentDiffusion(...)
        
        # Stage1: 不需要 LLM
        # Stage2: 需要 LLM（冻结或 LoRA）
        
    def forward(self, batch):
        if self.training_stage == 1:
            return self._forward_stage1(batch)
        else:
            return self._forward_stage2(batch)
    
    def _forward_stage1(self, batch):
        # 加载预保存的 Hidden States
        # 只计算 Diffusion Loss
        pass
    
    def _forward_stage2(self, batch):
        # Question → LLM → Question Hidden State
        # Question Hidden State → Diffusion → Steps Hidden State
        # Question Hidden State + Steps Hidden State → LLM → Answer
        pass
```

### 2.2 数据预处理模块

**文件路径**：`scripts/extract_hidden_states.py`

**核心功能**：
1. 加载训练数据
2. 对每个样本进行 LLM forward
3. 提取 Last Hidden States
4. 保存到本地文件

**数据流**：
```
原始数据 (question, steps, answer)
    ↓
Tokenizer + Embedding
    ↓
LLM Forward (output_hidden_states=True)
    ↓
提取 Last Hidden State (outputs.hidden_states[-1])
    ↓
保存到 .pt 文件
```

### 2.3 Hidden State Dataset

**文件路径**：`src/data/hidden_state_dataset.py`

**核心功能**：
1. 加载预保存的 Hidden States
2. 提供 DataLoader 接口
3. 支持随机采样和批处理

**接口**：
```python
class HiddenStateDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        # 加载预保存的 Hidden States
        self.question_hidden = torch.load(...)
        self.steps_hidden = torch.load(...)
        self.question_mask = torch.load(...)
        self.steps_mask = torch.load(...)
    
    def __getitem__(self, idx):
        return {
            'question_hidden': self.question_hidden[idx],
            'steps_hidden': self.steps_hidden[idx],
            'question_mask': self.question_mask[idx],
            'steps_mask': self.steps_mask[idx],
        }
```

### 2.4 Latent Diffusion 模块

**复用**：`src/modules/diffusion/latent_diffusion.py`

**关键修改**：
- **输入维度**：从 Embedding 改为 Hidden State（维度相同，但语义不同）
- **归一化**：需要重新估计 Hidden State 的统计量
- **推理步数**：从 128 步减少到 32 步

**核心方法**：
```python
class LatentDiffusion(nn.Module):
    def __init__(self, ...):
        # 与 difflar_fused 相同
        pass
    
    def forward(self, steps_hidden_states, condition_hidden_states, ...):
        # 输入是 Hidden States 而非 Embeddings
        # 其他逻辑相同
        pass
    
    def generate(self, condition_hidden_states, num_inference_steps=32, ...):
        # 推理步数减少到 32
        pass
```

---

## 三、数据流设计

### 3.1 Stage1 数据流

```
预保存的 Hidden States
    ├─> Question Hidden State（条件输入）
    │   ⚠️ Query Last Hidden State
    └─> Steps Hidden State（学习目标）
        ⚠️ GT Steps Hidden State
    ↓
HiddenStateDataset
    ↓
DataLoader (batch_size=64)
    ↓
LitDiffLaRHidden.forward()
    ├─> question_hidden: [B, L_q, H]  # ⚠️ 条件输入：Query Last Hidden State
    └─> steps_hidden: [B, L_s, H]      # ⚠️ 学习目标：GT Steps Hidden State
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage1a：基础去噪训练（前 50% epochs）                    │
├─────────────────────────────────────────────────────────┤
│ LatentDiffusion.forward()                              │
│ ├─> condition: question_hidden                        │
│ ├─> steps_embeds: steps_hidden                        │
│ ├─> 时间步采样：t ~ U[0, 1]（均匀分布）                │
│ └─> Diffusion Loss                                     │
│                                                         │
│ LatentDiffusion.generate()                             │
│ ├─> condition: question_hidden                        │
│ ├─> num_inference_steps: 10（快速训练）               │
│ └─> Generated Steps Hidden State                       │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Stage1b：全噪声生成训练（后 50% epochs）                │
├─────────────────────────────────────────────────────────┤
│ LatentDiffusion.forward()                              │
│ ├─> condition: question_hidden                        │
│ ├─> steps_embeds: steps_hidden                        │
│ ├─> 时间步采样：50% t ~ U[0.8, 1.0]（高噪声）          │
│ │                 50% t ~ U[0, 1]（正常）              │
│ └─> Diffusion Loss                                     │
│                                                         │
│ LatentDiffusion.generate()                             │
│ ├─> condition: question_hidden                        │
│ ├─> num_inference_steps: 20（接近推理）               │
│ └─> Generated Steps Hidden State                       │
└─────────────────────────────────────────────────────────┘
    ↓
Alignment Loss（显式对齐生成和 GT）
    ↓
反向传播（只更新 Diffusion 参数）
```

### 3.2 Stage2 数据流

```
原始数据 (question, steps, answer)
    ↓
Question → LLM Forward → Question Hidden State
    ↓
Question Hidden State → Diffusion → Generated Steps Hidden State
    ↓
Question Hidden State + Generated Steps Hidden State → LLM Forward → Answer
    ↓
计算 Loss（Diffusion + Alignment + Answer）
    ↓
反向传播（更新 Diffusion + LLM LoRA 参数）
```

### 3.3 推理数据流

```
Question (文本)
    ↓
Question → LLM Forward → Question Hidden State
    ↓
Question Hidden State → Diffusion (32步) → Steps Hidden State
    ↓
Question Hidden State + Steps Hidden State → LLM Forward → Answer
    ↓
Answer (文本)
```

---

## 四、关键设计决策

### 4.1 为什么 Stage1 不需要 LLM？

**原因**：
1. **训练效率**：不需要 LLM 前向传播，训练更快
2. **显存节省**：不需要加载 LLM，显存占用更小
3. **解耦训练**：可以独立优化 Diffusion 模型

**实现**：
- Stage1 只加载预保存的 Hidden States
- 不加载 LLM 模型
- 只训练 Diffusion 模型参数

### 4.2 为什么使用 Last Hidden State 而非 Embedding？

**原因**：
1. **连续性**：Last Hidden State 是连续的特征空间，更适合 Diffusion
2. **语义丰富**：经过多层 Transformer 处理，包含更丰富的语义信息
3. **训练稳定**：连续空间的梯度更稳定

**实现**：
- 在数据预处理阶段提取 Last Hidden State
- 保存到本地文件
- 训练时直接加载使用

### 4.3 为什么推理步数减少到 32 步？

**原因**：
1. **连续空间**：Last Hidden State 空间更连续，需要更少的去噪步数
2. **计算效率**：减少推理时间，提升用户体验
3. **质量保证**：通过实验验证，32 步的质量可接受

**实现**：
- 配置参数：`num_inference_steps: 32`
- 使用 Flow Matching 模式，支持更少的推理步数

---

## 五、与 DiffLaR Fused 的架构对比

| 维度 | DiffLaR Fused | DiffLaR Hidden |
|------|---------------|----------------|
| **Stage1 输入** | 实时计算的 Embedding | 预保存的 Hidden State |
| **Stage1 LLM** | 需要（冻结） | **不需要** |
| **Stage2 输入** | 实时计算的 Embedding | 实时计算的 Hidden State |
| **数据存储** | 无 | **预保存 Hidden States** |
| **推理步数** | 128 步 | **32 步** |
| **Denoiser 输入** | Embedding | Hidden State |

---

## 六、文件结构

```
src/
├── models/
│   ├── difflar_hidden.py          # 主模型类 LitDiffLaRHidden
│   └── model_base.py               # 基类（复用）
├── modules/
│   └── diffusion/
│       ├── latent_diffusion.py     # Latent Diffusion（复用，需适配）
│       ├── denoiser.py             # Denoiser（复用）
│       └── scheduler.py            # 噪声调度器（复用）
├── data/
│   └── hidden_state_dataset.py     # Hidden State 数据集（新建）
└── configs/
    └── models/
        └── difflar_hidden.yaml     # 配置文件（新建）

scripts/
└── extract_hidden_states.py       # 数据预处理脚本（新建）

data/
└── hidden_states/                  # 预保存的 Hidden States
    ├── train/
    │   ├── question_hidden_states.pt
    │   ├── steps_hidden_states.pt
    │   └── ...
    └── val/
        └── ...
```

---

*文档版本：v1.0*  
*创建时间：2026-01-12*


