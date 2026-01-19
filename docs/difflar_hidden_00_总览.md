# DiffLaR Hidden 算法设计文档总览

## 一、文档结构

本文档集将 DiffLaR Hidden 算法的技术设计拆分为多个专题文档，便于理解和查阅：

### 📚 文档列表

1. **[算法概述](./difflar_hidden_01_算法概述.md)**
   - 核心思想和创新点
   - 与 DiffLaR Fused 的对比
   - 算法优势分析
   - 流程概览

2. **[架构设计](./difflar_hidden_02_架构设计.md)**
   - 整体系统架构
   - 核心模块设计
   - 数据流设计
   - 关键设计决策

3. **[数据预处理](./difflar_hidden_03_数据预处理.md)**
   - Last Hidden State 提取
   - 数据存储格式
   - 数据加载方式
   - 性能优化

4. **[训练流程](./difflar_hidden_04_训练流程.md)**
   - Stage1：Diffusion 独立训练
   - Stage2：联合训练
   - 阶段切换机制
   - 训练监控

5. **[推理流程](./difflar_hidden_05_推理流程.md)**
   - 完整推理步骤
   - 推理优化（32步）
   - 性能分析
   - 使用示例

6. **[配置参数](./difflar_hidden_06_配置参数.md)**
   - 完整配置说明
   - 参数详解
   - 配置建议
   - 与 DiffLaR Fused 对比

---

## 二、快速导航

### 🚀 快速开始

**如果你是第一次了解这个算法**：
1. 先阅读 [算法概述](./difflar_hidden_01_算法概述.md) 了解核心思想
2. 然后阅读 [架构设计](./difflar_hidden_02_架构设计.md) 理解整体结构

**如果你要开始实现**：
1. 阅读 [数据预处理](./difflar_hidden_03_数据预处理.md) 了解数据准备
2. 阅读 [训练流程](./difflar_hidden_04_训练流程.md) 了解训练细节
3. 参考 [配置参数](./difflar_hidden_06_配置参数.md) 进行配置

**如果你要使用模型**：
1. 阅读 [推理流程](./difflar_hidden_05_推理流程.md) 了解推理过程
2. 参考 [配置参数](./difflar_hidden_06_配置参数.md) 进行配置

---

## 三、核心要点总结

### 3.1 核心创新

1. **从 Embedding 空间 → Last Hidden State 空间**
   - Last Hidden State 是连续空间，更适合 Diffusion 模型
   - 包含更丰富的语义信息

2. **Stage1 完全解耦**
   - 不需要 LLM 参与，训练更快
   - 显存占用更小
   - 可以快速迭代实验

3. **推理步数优化**
   - 从 128 步减少到 32 步
   - 推理速度提升约 4 倍

### 3.2 关键流程

```
数据预处理（一次性）
    ├─> 提取 Question Hidden State（必须预保存）
    └─> 提取 Steps Hidden State（必须预保存）
    ↓
Stage1：Diffusion 独立训练（不需要 LLM）
    ├─> Stage1a：基础去噪训练（前 50%）
    │   ├─> 正常时间步采样（t ~ U[0, 1]）
    │   └─> 10 步生成（快速训练）
    │
    └─> Stage1b：全噪声生成训练（后 50%）
        ├─> 强化高噪声采样（t ~ U[0.8, 1.0]）
        └─> 20 步生成（接近推理）
    ↓
Stage2：联合训练（需要 LLM）
    ├─> 实时计算 Question Hidden State
    └─> Diffusion 生成 Steps Hidden State
    ↓
推理（32步）
    ├─> 实时计算 Question Hidden State
    └─> Diffusion 从全噪声生成 Steps Hidden State
```

### 3.3 与 DiffLaR Fused 的主要区别

| 维度 | DiffLaR Fused | DiffLaR Hidden |
|------|---------------|----------------|
| **学习目标** | Embedding | **Last Hidden State** |
| **Stage1** | 需要 LLM（冻结） | **不需要 LLM** |
| **数据准备** | 实时计算 | **预保存** |
| **推理步数** | 128 步 | **32 步** |

---

## 四、实现检查清单

### 4.1 数据预处理

- [ ] 实现 `extract_hidden_states.py` 脚本
- [ ] 提取并保存 Question Hidden States
- [ ] 提取并保存 Steps Hidden States
- [ ] 保存 attention masks
- [ ] 保存元信息

### 4.2 模型实现

- [ ] 实现 `LitDiffLaRHidden` 主模型类
- [ ] 实现 `HiddenStateDataset` 数据集类
- [ ] 实现 Stage1 训练逻辑
- [ ] 实现 Stage2 训练逻辑
- [ ] 实现推理逻辑

### 4.3 训练流程

- [ ] 实现 Stage1 训练（只训练 Diffusion）
- [ ] 实现阶段自动切换
- [ ] 实现 Stage2 训练（联合训练）
- [ ] 实现 Loss 记录和可视化

### 4.4 配置和测试

- [ ] 创建配置文件 `difflar_hidden.yaml`
- [ ] 测试数据预处理流程
- [ ] 测试 Stage1 训练
- [ ] 测试 Stage2 训练
- [ ] 测试推理流程

---

## 五、关键问题讨论

### 5.1 Hidden State → LLM 输入

**问题**：LLM 的 `forward` 方法通常接受 `inputs_embeds`（Embedding），但我们现在有 Hidden State。

**解决方案**：
- **方案1**：直接使用 Hidden State（推荐，因为维度相同）
- **方案2**：通过投影层映射回 Embedding 空间
- **方案3**：修改 LLM 代码支持 Hidden State 输入

**建议**：先尝试方案1，如果不行再考虑方案2。

### 5.2 归一化参数

**问题**：Hidden State 的分布与 Embedding 不同，需要重新估计归一化参数。

**解决方案**：
- 在 Stage1 训练前，从预保存的 Hidden States 估计统计量
- 使用与 DiffLaR Fused 相同的归一化方法

### 5.3 固定长度处理

**问题**：不同样本的 Hidden State 长度不同。

**解决方案**：
- Question Hidden State：padding/truncate 到 `max_condition_length`
- Steps Hidden State：padding/truncate 到 `max_latent_length`
- 保存 attention mask 用于后续训练

---

## 六、下一步行动

1. **确认设计**：与团队讨论并确认设计文档
2. **开始实现**：按照文档逐步实现各个模块
3. **测试验证**：每个模块实现后进行测试
4. **实验对比**：与 DiffLaR Fused 进行对比实验

---

## 七、文档维护

- **版本**：v1.0
- **创建时间**：2026-01-12
- **最后更新**：2026-01-12
- **维护者**：开发团队

**更新日志**：
- v1.0 (2026-01-12)：初始版本，完成所有6个专题文档

---

*如有问题或建议，请及时反馈！*


