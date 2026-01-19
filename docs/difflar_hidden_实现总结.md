# DiffLaR Hidden 实现总结

## 一、已实现的功能

### 1.1 数据预处理模块 ✅

**文件**: `scripts/extract_hidden_states.py`

- ✅ 提取 Question 的 Last Hidden State
- ✅ 提取 Steps 的 Last Hidden State
- ✅ 保存 attention masks
- ✅ 保存元信息（metadata.json）
- ✅ 支持批量处理
- ✅ 显存优化（及时转移到 CPU）

### 1.2 数据集模块 ✅

**文件**: 
- `src/data/hidden_state_dataset.py`
- `src/datasets/hidden_state_datamodule.py`

- ✅ `HiddenStateDataset`：加载预保存的 Hidden States
- ✅ `HiddenStateDataModule`：PyTorch Lightning DataModule
- ✅ 支持 train/val/test 分割
- ✅ 数据一致性验证

### 1.3 主模型 ✅

**文件**: `src/models/difflar_hidden.py`

#### Stage1 训练 ✅
- ✅ 使用预保存的 Hidden States（Question + Steps）
- ✅ Stage1a：基础去噪训练（10 步生成）
- ✅ Stage1b：全噪声生成训练（20 步生成）
- ✅ Diffusion Loss + Alignment Loss
- ✅ Self-Conditioning 支持
- ✅ 不需要 LLM（完全解耦）

#### Stage2 训练 ✅
- ✅ 实时计算 Question Hidden State
- ✅ Diffusion 生成 Steps Hidden State
- ✅ Answer Loss（端到端优化）
- ✅ Diffusion Loss + Alignment Loss + Answer Loss
- ✅ 联合训练 Diffusion + LLM（LoRA）

#### 推理 ✅
- ✅ 32 步推理（优化）
- ✅ 支持 CFG（可选）
- ✅ Self-Conditioning
- ✅ 返回 Steps Hidden State（可选）

#### 其他功能 ✅
- ✅ 阶段自动切换（Stage1 → Stage2）
- ✅ Loss 记录和可视化
- ✅ 归一化参数估计
- ✅ 固定长度处理（padding/truncate）

### 1.4 配置文件 ✅

**文件**: `src/configs/models/difflar_hidden.yaml`

- ✅ 完整的配置参数
- ✅ Stage1a/Stage1b 配置
- ✅ 推理步数优化（32 步）
- ✅ 损失权重配置

## 二、核心设计实现

### 2.1 两阶段训练策略

```
Stage1: Diffusion 独立训练
  ├─ 输入：预保存的 Hidden States
  ├─ 训练：只训练 Diffusion 模型
  └─ 特点：不需要 LLM，训练快速

Stage2: 联合训练
  ├─ 输入：原始数据（question, steps, answer）
  ├─ 训练：Diffusion + LLM（LoRA）
  └─ 特点：端到端优化
```

### 2.2 Hidden State 处理

- **Question Hidden State**：作为 Diffusion 的条件输入
- **Steps Hidden State**：作为 Diffusion 的学习目标
- **固定长度**：通过 padding/truncate 处理变长序列
- **归一化**：从训练数据估计统计量

### 2.3 Loss 计算

- **Diffusion Loss**：学习去噪能力
- **Alignment Loss**：对齐生成和 GT（MSE + Cosine）
- **Answer Loss**：端到端优化（仅 Stage2）

## 三、与 DiffLaR Fused 的主要区别

| 维度 | DiffLaR Fused | DiffLaR Hidden |
|------|---------------|----------------|
| **学习目标** | Embedding | **Last Hidden State** |
| **Stage1 输入** | 实时计算的 Embedding | **预保存的 Hidden State** |
| **Stage1 LLM** | 需要（冻结） | **不需要** |
| **数据准备** | 实时计算 | **预保存** |
| **推理步数** | 128 步 | **32 步** |
| **训练速度** | 较慢 | **更快（Stage1）** |

## 四、使用流程

### 4.1 数据预处理

```bash
python scripts/extract_hidden_states.py \
    --model_path models/llms/Llama-3.2-1B-Instruct \
    --data_path datasets/text_reasoning/your_dataset/train.json \
    --output_dir data/hidden_states/train \
    --batch_size 32
```

### 4.2 训练

```bash
# Stage1: 使用 HiddenStateDataModule
python run.py --config src/configs/models/difflar_hidden.yaml

# Stage2: 自动切换到 QSADataModule（需要在代码中实现）
```

### 4.3 推理

```python
model = LitDiffLaRHidden.load_from_checkpoint("checkpoints/best.ckpt")
answers, n_latent = model.latent_generate(questions)
```

## 五、待完善的功能

### 5.1 数据集自动切换

当前实现中，Stage1 和 Stage2 需要使用不同的 DataModule。需要在训练脚本中实现自动切换逻辑。

**建议实现**：
- 在 `on_train_epoch_start` 中检查阶段
- Stage1：使用 `HiddenStateDataModule`
- Stage2：切换到 `QSADataModule`

### 5.2 Stage1b 高噪声采样

当前实现中，Stage1b 的高噪声采样（t ~ U[0.8, 1.0]）需要在 `LatentDiffusion._forward_flow_matching` 中添加支持。

**建议实现**：
- 在 `LatentDiffusion.forward` 中添加 `t_min` 和 `t_max` 参数
- 在 `_forward_flow_matching` 中使用这些参数控制时间采样范围

### 5.3 验证和测试

需要添加：
- 验证集上的评估逻辑
- 测试集上的评估逻辑
- 与 DiffLaR Fused 的对比实验

## 六、文件结构

```
colar/
├── scripts/
│   └── extract_hidden_states.py          # 数据预处理脚本
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── hidden_state_dataset.py       # Hidden State 数据集
│   ├── datasets/
│   │   └── hidden_state_datamodule.py    # DataModule
│   ├── models/
│   │   └── difflar_hidden.py             # 主模型
│   └── configs/
│       └── models/
│           └── difflar_hidden.yaml       # 配置文件
└── docs/
    ├── difflar_hidden_使用指南.md
    └── difflar_hidden_实现总结.md
```

## 七、下一步工作

1. **实现数据集自动切换**：在训练脚本中添加 Stage1/Stage2 的数据集切换逻辑
2. **完善高噪声采样**：在 LatentDiffusion 中添加时间采样范围控制
3. **添加评估逻辑**：实现验证和测试的评估代码
4. **实验验证**：与 DiffLaR Fused 进行对比实验
5. **性能优化**：进一步优化训练和推理速度

## 八、注意事项

1. **数据一致性**：确保 Question 和 Steps 的 Hidden States 来自同一个模型和配置
2. **归一化参数**：Stage1 训练前需要估计 Hidden State 的归一化参数
3. **显存管理**：Stage1 虽然不需要 LLM，但 Hidden States 可能占用较多显存
4. **阶段切换**：确保 Stage1 训练完成后才切换到 Stage2



