# DiffLaR Hidden 使用指南

## 一、快速开始

### 1.1 数据预处理

首先需要提取并保存 Last Hidden States：

```bash
python scripts/extract_hidden_states.py \
    --model_path models/llms/Llama-3.2-1B-Instruct \
    --data_path datasets/text_reasoning/your_dataset/train.json \
    --output_dir data/hidden_states/train \
    --batch_size 32 \
    --max_question_length 512 \
    --max_steps_length 512 \
    --device cuda
```

对验证集和测试集也执行相同的操作：

```bash
# 验证集
python scripts/extract_hidden_states.py \
    --model_path models/llms/Llama-3.2-1B-Instruct \
    --data_path datasets/text_reasoning/your_dataset/val.json \
    --output_dir data/hidden_states/val \
    --batch_size 32 \
    --device cuda

# 测试集（可选）
python scripts/extract_hidden_states.py \
    --model_path models/llms/Llama-3.2-1B-Instruct \
    --data_path datasets/text_reasoning/your_dataset/test.json \
    --output_dir data/hidden_states/test \
    --batch_size 32 \
    --device cuda
```

### 1.2 训练

#### Stage1 训练（使用 Hidden State Dataset）

修改配置文件，使用 `HiddenStateDataModule`：

```yaml
# config.yaml
datamodule:
  target: src.datasets.hidden_state_datamodule.HiddenStateDataModule
  hidden_states_dir: data/hidden_states
```

然后开始训练：

```bash
python run.py --config src/configs/models/difflar_hidden.yaml
```

#### Stage2 训练（使用原始 QSA Dataset）

Stage2 会自动切换到原始数据集。确保配置文件中有 QSA DataModule 的配置。

### 1.3 推理

```python
from src.models.difflar_hidden import LitDiffLaRHidden

# 加载模型
model = LitDiffLaRHidden.load_from_checkpoint("checkpoints/best.ckpt")
model.eval()

# 推理
questions = ["What is 2+2?"]
answers, n_latent = model.latent_generate(questions)

# 解码
answer_texts = model.tokenizer.batch_decode(answers, skip_special_tokens=True)
print(answer_texts)
```

## 二、关键文件说明

### 2.1 数据预处理

- **`scripts/extract_hidden_states.py`**：提取并保存 Last Hidden States

### 2.2 数据集

- **`src/data/hidden_state_dataset.py`**：加载预保存的 Hidden States
- **`src/datasets/hidden_state_datamodule.py`**：Stage1 训练使用的 DataModule

### 2.3 模型

- **`src/models/difflar_hidden.py`**：DiffLaR Hidden 主模型实现

### 2.4 配置

- **`src/configs/models/difflar_hidden.yaml`**：模型配置文件

## 三、训练流程

### 3.1 Stage1：Diffusion 独立训练

- **输入**：预保存的 Hidden States（Question + Steps）
- **目标**：训练 Diffusion 模型学会生成 Steps Hidden State
- **特点**：不需要 LLM，训练速度快

### 3.2 Stage2：联合训练

- **输入**：原始数据（question, steps, answer）
- **目标**：端到端优化 Diffusion + LLM
- **特点**：需要 LLM，训练速度较慢

## 四、配置参数说明

详见 `docs/difflar_hidden_06_配置参数.md`

## 五、常见问题

### Q1: Stage1 训练时显存不足？

**A**: 减少 `batch_size` 或使用梯度累积。

### Q2: 如何切换 Stage1 和 Stage2 的数据集？

**A**: Stage1 使用 `HiddenStateDataModule`，Stage2 会自动切换到原始 `QSADataModule`。需要在代码中实现自动切换逻辑。

### Q3: 推理步数可以调整吗？

**A**: 可以，修改配置文件中的 `num_inference_steps` 参数。

## 六、性能优化建议

1. **数据预处理**：使用多进程加速（`num_workers > 0`）
2. **训练**：Stage1 可以使用更大的 batch_size（不需要 LLM）
3. **推理**：32 步已经足够，可以尝试更少的步数（如 16 步）以进一步提升速度



