# DiffLaR Hidden 配置参数

## 一、配置文件结构

```yaml
seed: ~

model:
  target: src.models.difflar_hidden.LitDiffLaRHidden
  model_kwargs:
    model_id: Llama-3.2-1B-Instruct
    sft_method: difflar_hidden
    chat_template: False

    do_lora: True
    lora_config:
      r: 128
      lora_alpha: 32

    difflar_config:
      # ═══ Diffusion模型配置 ═══
      max_latent_length: 256
      max_condition_length: 128
      num_timesteps: 1000
      num_inference_steps: 32          # 关键：从 128 减少到 32
      train_inference_steps: 10
      noise_schedule: linear
      
      # 去噪网络配置
      denoiser_layers: 6
      denoiser_heads: 8
      dropout: 0.1
      
      # 损失权重
      diffusion_loss_weight: 1.0     # Diffusion Loss 权重（Stage1 和 Stage2）
      alignment_loss_weight: 1.0      # Alignment Loss 权重（Stage1 和 Stage2）
      answer_loss_weight: 5.0         # Answer Loss 权重（仅 Stage2）
      
      # CFG配置
      use_cfg: False
      cfg_scale: 1.5
      
      # 数据归一化
      normalize_latent: True
      
      # 预测目标
      prediction_type: flow
      clamp_value: 3.0
      
      # 采样器类型
      sampler_type: euler
      ddim_eta: 0.0
      
      # ═══ 两阶段训练配置 ═══
      stage1_epochs: 20              # Stage1 总 epoch 数
      stage1a_ratio: 0.5            # Stage1a 占比（前 50%）
      stage1b_ratio: 0.5            # Stage1b 占比（后 50%）
      
      # ═══ Stage1a 配置（基础去噪训练） ═══
      stage1a_train_inference_steps: 10      # 快速训练
      
      # ═══ Stage1b 配置（全噪声生成训练） ═══
      stage1b_train_inference_steps: 20      # 接近推理（32步）
      stage1b_high_noise_ratio: 0.5          # 高噪声采样比例
      
      # ═══ Self-Conditioning配置 ═══
      stage1_self_cond_prob: 0.5
      stage2_self_cond_prob: 1.0

    answer_generation_config:
      max_new_tokens: 16
      do_sample: True
      top_p: 0.9
      temperature: 1.0

  training_kwargs:
    optimizer:
      target: torch.optim.AdamW
      lr: 1e-4
      weight_decay: 0.01
    use_scheduler: False
    scheduler:
      target: constant_schedule_with_warmup
      warmup_steps: 1000

trainer:
  max_epochs: 20

dataloader:
  batch_size: 64
  val_batch_size: 32
  num_workers: 8
  pin_memory: True
  persistent_workers: True
```

---

## 二、核心配置参数详解

### 2.1 Diffusion 模型配置

#### `max_latent_length`
- **类型**：`int`
- **默认值**：`256`
- **说明**：Steps Hidden State 的最大长度
- **用途**：所有 Steps Hidden State 都会被 padding/truncate 到这个长度

#### `max_condition_length`
- **类型**：`int`
- **默认值**：`128`
- **说明**：Question Hidden State 的最大长度
- **用途**：所有 Question Hidden State 都会被 padding/truncate 到这个长度

#### `num_timesteps`
- **类型**：`int`
- **默认值**：`1000`
- **说明**：扩散过程的总时间步数
- **用途**：定义扩散过程的离散化程度

#### `num_inference_steps` ⭐
- **类型**：`int`
- **默认值**：`32`（从 128 减少）
- **说明**：**推理时的去噪步数**（关键优化）
- **用途**：控制推理质量和速度的平衡
- **建议**：
  - 32 步：快速推理，质量可接受
  - 64 步：平衡质量和速度
  - 128 步：高质量，但速度较慢

#### `train_inference_steps`
- **类型**：`int`
- **默认值**：`10`
- **说明**：训练时 Diffusion 生成使用的步数
- **用途**：训练时快速生成，节省计算资源

#### `noise_schedule`
- **类型**：`str`
- **可选值**：`"linear"`, `"cosine"`
- **默认值**：`"linear"`
- **说明**：噪声调度类型
- **建议**：使用 `"linear"`（Flow Matching 模式）

### 2.2 去噪网络配置

#### `denoiser_layers`
- **类型**：`int`
- **默认值**：`6`
- **说明**：Denoiser Transformer 的层数
- **建议**：6-12 层

#### `denoiser_heads`
- **类型**：`int`
- **默认值**：`8`
- **说明**：每个注意力层的头数
- **建议**：8-16 头

#### `dropout`
- **类型**：`float`
- **默认值**：`0.1`
- **说明**：Dropout 率
- **建议**：0.1-0.2

### 2.3 损失权重配置

#### `diffusion_loss_weight`
- **类型**：`float`
- **默认值**：`1.0`
- **说明**：Diffusion Loss 的权重
- **用途**：控制 Diffusion 模型的学习强度

#### `alignment_loss_weight`
- **类型**：`float`
- **默认值**：`1.0`
- **说明**：Alignment Loss 的权重（Stage2）
- **用途**：控制生成 Hidden State 与 GT 的对齐程度

#### `answer_loss_weight`
- **类型**：`float`
- **默认值**：`5.0`
- **说明**：Answer Loss 的权重（Stage2）
- **用途**：控制端到端优化的强度

### 2.4 CFG 配置

#### `use_cfg`
- **类型**：`bool`
- **默认值**：`False`
- **说明**：是否使用 Classifier-Free Guidance
- **注意**：CFG 需要每步 2 次 forward，计算成本较高

#### `cfg_scale`
- **类型**：`float`
- **默认值**：`1.5`
- **说明**：CFG 的强度（如果启用）
- **建议**：1.0-2.0

### 2.5 数据归一化配置

#### `normalize_latent`
- **类型**：`bool`
- **默认值**：`True`
- **说明**：是否对 Hidden State 进行归一化
- **建议**：**必须启用**，确保训练-推理分布一致

### 2.6 预测目标配置

#### `prediction_type`
- **类型**：`str`
- **可选值**：`"epsilon"`, `"x0"`, `"flow"`
- **默认值**：`"flow"`
- **说明**：预测目标类型
- **建议**：使用 `"flow"`（更稳定，支持更少的推理步数）

#### `sampler_type`
- **类型**：`str`
- **可选值**：`"ddpm"`, `"ddim"`, `"euler"`
- **默认值**：`"euler"`
- **说明**：采样器类型
- **建议**：`"flow"` 模式使用 `"euler"`

#### `clamp_value`
- **类型**：`float`
- **默认值**：`3.0`
- **说明**：数值截断阈值
- **用途**：防止数值爆炸

### 2.7 两阶段训练配置

#### `stage1_epochs`
- **类型**：`int`
- **默认值**：`10`
- **说明**：Stage1 训练的 epoch 数
- **建议**：5-15 epochs

### 2.8 Self-Conditioning 配置

#### `stage1_self_cond_prob`
- **类型**：`float`
- **默认值**：`0.5`
- **说明**：Stage1 时 Self-Conditioning 的启用概率
- **范围**：`[0.0, 1.0]`

#### `stage2_self_cond_prob`
- **类型**：`float`
- **默认值**：`1.0`
- **说明**：Stage2 时 Self-Conditioning 的启用概率（通常为 1.0，始终启用）

---

## 三、数据预处理配置

### 3.1 提取 Hidden States 配置

```python
# scripts/extract_hidden_states.py 的参数

extraction_config = {
    "batch_size": 32,           # 批处理大小
    "max_length": 512,          # 最大序列长度
    "output_dir": "data/hidden_states",  # 输出目录
}
```

---

## 四、训练配置

### 4.1 优化器配置

```yaml
training_kwargs:
  optimizer:
    target: torch.optim.AdamW
    lr: 1e-4                    # 学习率
    weight_decay: 0.01          # 权重衰减
```

**建议学习率**：
- Stage1：`1e-4` - `5e-4`
- Stage2：`1e-4` - `2e-4`

### 4.2 调度器配置

```yaml
training_kwargs:
  use_scheduler: False          # 是否使用学习率调度器
  scheduler:
    target: constant_schedule_with_warmup
    warmup_steps: 1000
```

### 4.3 DataLoader 配置

```yaml
dataloader:
  batch_size: 64               # 训练 batch size
  val_batch_size: 32          # 验证 batch size
  num_workers: 8               # 数据加载进程数
  pin_memory: True            # 是否使用 pin_memory
  persistent_workers: True    # 是否保持 worker 进程
```

---

## 五、推理配置

### 5.1 Answer 生成配置

```yaml
answer_generation_config:
  max_new_tokens: 16          # 最大生成 token 数
  do_sample: True            # 是否采样
  top_p: 0.9                 # Nucleus sampling
  temperature: 1.0           # 温度参数
```

---

## 六、配置对比：DiffLaR Fused vs DiffLaR Hidden

| 配置项 | DiffLaR Fused | DiffLaR Hidden | 说明 |
|--------|---------------|----------------|------|
| `num_inference_steps` | 128 | **32** | 关键优化 |
| `max_condition_length` | 无 | **128** | 新增：Question 长度限制 |
| `stage1_epochs` | 5 | **10** | 建议更长（独立训练） |
| 数据输入 | 实时 Embedding | **预保存 Hidden State** | 关键差异 |

---

## 七、配置建议

### 7.1 快速实验配置

```yaml
difflar_config:
  num_inference_steps: 32
  train_inference_steps: 5
  stage1_epochs: 5
  denoiser_layers: 4
  denoiser_heads: 4
```

### 7.2 高质量配置

```yaml
difflar_config:
  num_inference_steps: 64
  train_inference_steps: 10
  stage1_epochs: 15
  denoiser_layers: 8
  denoiser_heads: 12
```

### 7.3 平衡配置（推荐）

```yaml
difflar_config:
  num_inference_steps: 32
  train_inference_steps: 10
  stage1_epochs: 10
  denoiser_layers: 6
  denoiser_heads: 8
```

---

## 八、环境变量配置

### 8.1 数据路径

```bash
export HIDDEN_STATES_DIR="data/hidden_states"
export MODEL_PATH="models/llms/Llama-3.2-1B-Instruct"
```

### 8.2 显存配置

```bash
# 如果显存不足，可以减少 batch_size
export BATCH_SIZE=32
```

---

*文档版本：v1.0*  
*创建时间：2026-01-12*


