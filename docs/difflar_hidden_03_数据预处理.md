# DiffLaR Hidden 数据预处理

## 一、概述

数据预处理是 DiffLaR Hidden 算法的关键步骤，需要在训练前一次性完成。主要任务是提取并保存 **Question 和 Steps 对应的 Last Hidden States**，供 Stage1 训练使用。

**关键点**：
- **Question Hidden State**：作为 Diffusion 的条件输入（condition）
- **Steps Hidden State**：作为 Diffusion 的学习目标（target）
- **两者都需要预保存**：因为 Stage1 训练不需要 LLM，无法实时计算

---

## 二、Last Hidden State 提取

### 2.1 什么是 Last Hidden State？

**定义**：LLM 最后一层 Transformer 的输出，是经过所有层处理后的连续特征表示。

**获取方式**：
```python
outputs = llm.forward(
    inputs_embeds=embeddings,
    output_hidden_states=True  # 关键：需要返回 hidden states
)
last_hidden_state = outputs.hidden_states[-1]  # [B, L, H]
```

**特点**：
- **维度**：`[batch_size, seq_length, hidden_size]`
- **连续性**：是连续的特征空间，非离散的 token
- **语义丰富**：包含经过多层 Transformer 处理的语义信息

### 2.2 为什么使用 Last Hidden State？

| 特性 | Embedding | Last Hidden State |
|------|-----------|-------------------|
| **空间类型** | 离散的高维向量空间 | 连续的特征空间 |
| **语义信息** | 基础 token 表示 | 经过多层处理的语义表示 |
| **适合 Diffusion** | ❌ 不太适合 | ✅ 更适合 |
| **梯度稳定性** | 一般 | 更好 |

---

## 三、数据提取流程

### 3.1 完整流程

```
┌─────────────────────────────────────────────────────────┐
│           数据预处理完整流程                               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. 加载训练数据                                          │
│     └─> question, steps, answer (文本)                     │
│                                                          │
│  2. 对每个样本：                                          │
│     ├─> Question → Tokenizer → Embedding                │
│     │   └─> LLM Forward → Question Hidden State        │
│     │                                                      │
│     └─> Steps → Tokenizer → Embedding                   │
│         └─> LLM Forward → Steps Hidden State            │
│                                                          │
│  3. 收集所有 Hidden States                                │
│     ├─> question_hidden_states: [N, L_q, H]             │
│     │   ⚠️ 关键：Question Hidden State 也需要预保存！      │
│     │   （Stage1 需要作为 Diffusion 的条件输入）          │
│     ├─> steps_hidden_states: [N, L_s, H]               │
│     │   （Stage1 需要作为 Diffusion 的学习目标）          │
│     ├─> question_attention_mask: [N, L_q]               │
│     └─> steps_attention_mask: [N, L_s]                 │
│                                                          │
│  4. 保存到本地文件                                        │
│     └─> data/hidden_states/train/*.pt                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 代码实现框架

```python
# scripts/extract_hidden_states.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def extract_hidden_states(
    model_path: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 32,
    max_length: int = 512,
):
    """
    提取并保存 Last Hidden States
    
    Args:
        model_path: LLM 模型路径
        data_path: 训练数据路径
        output_dir: 输出目录
        batch_size: 批处理大小
        max_length: 最大序列长度
    """
    # 1. 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 2. 加载训练数据
    dataset = load_dataset(data_path)  # 自定义数据加载函数
    
    # 3. 初始化存储容器
    all_question_hidden = []
    all_steps_hidden = []
    all_question_mask = []
    all_steps_mask = []
    
    # 4. 批量处理
    for batch in tqdm(DataLoader(dataset, batch_size=batch_size)):
        questions = batch["question"]
        steps = batch["steps"]
        
        # 4.1 处理 Question
        question_inputs = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        question_embeds = model.get_input_embeddings()(question_inputs.input_ids)
        
        with torch.no_grad():
            question_outputs = model.forward(
                inputs_embeds=question_embeds,
                attention_mask=question_inputs.attention_mask,
                output_hidden_states=True,
            )
        
        question_hidden = question_outputs.hidden_states[-1]  # [B, L_q, H]
        question_mask = question_inputs.attention_mask  # [B, L_q]
        
        # 4.2 处理 Steps
        steps_inputs = tokenizer(
            steps,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        steps_embeds = model.get_input_embeddings()(steps_inputs.input_ids)
        
        with torch.no_grad():
            steps_outputs = model.forward(
                inputs_embeds=steps_embeds,
                attention_mask=steps_inputs.attention_mask,
                output_hidden_states=True,
            )
        
        steps_hidden = steps_outputs.hidden_states[-1]  # [B, L_s, H]
        steps_mask = steps_inputs.attention_mask  # [B, L_s]
        
        # 4.3 保存到 CPU（节省显存）
        all_question_hidden.append(question_hidden.cpu())
        all_steps_hidden.append(steps_hidden.cpu())
        all_question_mask.append(question_mask.cpu())
        all_steps_mask.append(steps_mask.cpu())
    
    # 5. 合并并保存
    all_question_hidden = torch.cat(all_question_hidden, dim=0)  # [N, L_q, H]
    all_steps_hidden = torch.cat(all_steps_hidden, dim=0)  # [N, L_s, H]
    all_question_mask = torch.cat(all_question_mask, dim=0)  # [N, L_q]
    all_steps_mask = torch.cat(all_steps_mask, dim=0)  # [N, L_s]
    
    # 6. 保存到文件
    os.makedirs(output_dir, exist_ok=True)
    torch.save(all_question_hidden, os.path.join(output_dir, "question_hidden_states.pt"))
    torch.save(all_steps_hidden, os.path.join(output_dir, "steps_hidden_states.pt"))
    torch.save(all_question_mask, os.path.join(output_dir, "question_attention_mask.pt"))
    torch.save(all_steps_mask, os.path.join(output_dir, "steps_attention_mask.pt"))
    
    # 7. 保存元信息
    metadata = {
        "num_samples": len(dataset),
        "hidden_size": all_question_hidden.shape[-1],
        "max_question_length": all_question_hidden.shape[1],
        "max_steps_length": all_steps_hidden.shape[1],
        "model_path": model_path,
    }
    import json
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Hidden states saved to {output_dir}")
    print(f"Total samples: {metadata['num_samples']}")
```

---

## 四、数据存储格式

### 4.1 文件结构

```
data/
└── hidden_states/
    ├── train/
    │   ├── question_hidden_states.pt      # [N_train, L_q, H]
    │   ├── steps_hidden_states.pt         # [N_train, L_s, H]
    │   ├── question_attention_mask.pt      # [N_train, L_q]
    │   ├── steps_attention_mask.pt         # [N_train, L_s]
    │   └── metadata.json                   # 元信息
    └── val/
        ├── question_hidden_states.pt       # [N_val, L_q, H]
        ├── steps_hidden_states.pt          # [N_val, L_s, H]
        ├── question_attention_mask.pt      # [N_val, L_q]
        ├── steps_attention_mask.pt         # [N_val, L_s]
        └── metadata.json                   # 元信息
```

### 4.2 数据格式说明

| 文件 | 形状 | 数据类型 | 说明 |
|------|------|---------|------|
| `question_hidden_states.pt` | `[N, L_q, H]` | `torch.FloatTensor` | Question 的 Last Hidden States |
| `steps_hidden_states.pt` | `[N, L_s, H]` | `torch.FloatTensor` | Steps 的 Last Hidden States |
| `question_attention_mask.pt` | `[N, L_q]` | `torch.LongTensor` | Question 的 attention mask |
| `steps_attention_mask.pt` | `[N, L_s]` | `torch.LongTensor` | Steps 的 attention mask |
| `metadata.json` | - | JSON | 数据元信息 |

### 4.3 元信息格式

```json
{
  "num_samples": 10000,
  "hidden_size": 2048,
  "max_question_length": 128,
  "max_steps_length": 256,
  "model_path": "models/llms/Llama-3.2-1B-Instruct",
  "extraction_date": "2026-01-12",
  "extraction_config": {
    "batch_size": 32,
    "max_length": 512
  }
}
```

---

## 五、数据加载

### 5.1 HiddenStateDataset 实现

```python
# src/data/hidden_state_dataset.py

import torch
from torch.utils.data import Dataset
import os
import json

class HiddenStateDataset(Dataset):
    """
    加载预保存的 Hidden States 数据集
    
    注意：Question 和 Steps 的 Hidden States 都需要预保存
    - Question Hidden State：作为 Diffusion 的条件输入
    - Steps Hidden State：作为 Diffusion 的学习目标
    """
    
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: 数据根目录（包含 train/ 和 val/ 子目录）
            split: "train" 或 "val"
        """
        self.data_dir = os.path.join(data_dir, split)
        
        # 加载数据（Question 和 Steps 都需要）
        self.question_hidden = torch.load(
            os.path.join(self.data_dir, "question_hidden_states.pt")
        )  # ⚠️ 必须：Stage1 需要作为条件输入
        self.steps_hidden = torch.load(
            os.path.join(self.data_dir, "steps_hidden_states.pt")
        )  # ⚠️ 必须：Stage1 需要作为学习目标
        self.question_mask = torch.load(
            os.path.join(self.data_dir, "question_attention_mask.pt")
        )
        self.steps_mask = torch.load(
            os.path.join(self.data_dir, "steps_attention_mask.pt")
        )
        
        # 加载元信息
        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        
        # 验证数据一致性
        assert len(self.question_hidden) == len(self.steps_hidden)
        assert len(self.question_hidden) == len(self.question_mask)
        assert len(self.steps_hidden) == len(self.steps_mask)
        
        self.num_samples = len(self.question_hidden)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "question_hidden": self.question_hidden[idx],  # [L_q, H]
            "steps_hidden": self.steps_hidden[idx],  # [L_s, H]
            "question_mask": self.question_mask[idx],  # [L_q]
            "steps_mask": self.steps_mask[idx],  # [L_s]
        }
```

### 5.2 DataLoader 配置

```python
from torch.utils.data import DataLoader

# Stage1 训练时使用
train_dataset = HiddenStateDataset(data_dir="data/hidden_states", split="train")
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

---

## 六、关键注意事项

### 6.1 显存管理

**问题**：提取 Hidden States 时可能显存不足

**解决方案**：
1. **批量处理**：使用合适的 batch_size（如 32）
2. **及时释放**：每批处理完后立即转移到 CPU
3. **梯度禁用**：使用 `torch.no_grad()` 禁用梯度计算

```python
with torch.no_grad():
    outputs = model.forward(...)
    hidden = outputs.hidden_states[-1].cpu()  # 立即转移到 CPU
```

### 6.2 序列长度处理

**问题**：不同样本的序列长度不同

**解决方案**：
1. **Padding**：使用 tokenizer 的 padding 功能
2. **Truncation**：设置 max_length 截断过长序列
3. **保存 Mask**：保存 attention_mask 用于后续训练

### 6.3 数据一致性

**验证**：
- 确保 question 和 steps 的样本数量一致
- 确保 mask 的维度与 hidden states 一致
- 保存元信息以便后续验证

---

## 七、使用示例

### 7.1 提取 Hidden States

```bash
python scripts/extract_hidden_states.py \
    --model_path models/llms/Llama-3.2-1B-Instruct \
    --data_path data/train.json \
    --output_dir data/hidden_states/train \
    --batch_size 32 \
    --max_length 512
```

### 7.2 验证数据

```python
# 验证加载的数据
dataset = HiddenStateDataset("data/hidden_states", split="train")
sample = dataset[0]

print(f"Question hidden shape: {sample['question_hidden'].shape}")
print(f"Steps hidden shape: {sample['steps_hidden'].shape}")
print(f"Question mask shape: {sample['question_mask'].shape}")
print(f"Steps mask shape: {sample['steps_mask'].shape}")
```

---

## 八、性能优化

### 8.1 并行处理

- 使用多进程 DataLoader（`num_workers > 0`）
- 使用 `pin_memory=True` 加速 GPU 传输

### 8.2 数据压缩

- 可以考虑使用半精度（`float16`）保存，减少存储空间
- 使用压缩格式（如 `torch.save(..., _use_new_zipfile_serialization=True)`）

### 8.3 增量提取

- 支持断点续传，避免重复提取
- 保存提取进度，支持增量更新

---

*文档版本：v1.0*  
*创建时间：2026-01-12*


