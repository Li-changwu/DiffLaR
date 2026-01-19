"""
DiffLaR Hidden 模型实现

核心改进：
1. 从 Embedding 空间 → Last Hidden State 空间
2. Stage1 完全解耦：不需要 LLM，使用预保存的 Hidden States
3. Stage2 联合训练：Diffusion + LLM
4. 推理步数优化：从 128 步减少到 32 步
"""

import os
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Optional, Dict

from .model_base import LitCoTModelBase
from ..modules.diffusion import LatentDiffusion
from ..utils.utils import get_position_ids_from_attention_mask


class LitDiffLaRHidden(LitCoTModelBase):
    """
    DiffLaR Hidden: 使用 Last Hidden State 空间的 Diffusion 模型
    
    核心特点：
    1. Stage1：Diffusion 独立训练（不需要 LLM）
    2. Stage2：联合训练（Diffusion + LLM）
    3. 使用预保存的 Hidden States 进行 Stage1 训练
    """

    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            all_config=all_config,
        )

        difflar_config = model_kwargs.difflar_config
        # Stage1 完全解耦（load_llm=false）时，self.llm 为 None，此时必须从配置显式提供 hidden_size
        if self.llm is not None:
            self.hidden_size = self.llm.config.hidden_size
        else:
            self.hidden_size = difflar_config.get("hidden_size")
            if self.hidden_size is None:
                raise ValueError(
                    "load_llm=false 时需要在 difflar_config.hidden_size 显式指定隐藏维度（例如 Llama-3.2-1B 为 2048）"
                )

        # Latent Diffusion模型
        self.latent_diffusion = LatentDiffusion(
            hidden_size=self.hidden_size,
            num_timesteps=difflar_config.get("num_timesteps", 1000),
            num_layers=difflar_config.get("denoiser_layers", 6),
            num_heads=difflar_config.get("denoiser_heads", 8),
            max_latent_length=difflar_config.get("max_latent_length", 256),
            schedule_type=difflar_config.get("noise_schedule", "linear"),
            dropout=difflar_config.get("dropout", 0.1),
            normalize_latent=difflar_config.get("normalize_latent", True),
            prediction_type=difflar_config.get("prediction_type", "flow"),
            sampler_type=difflar_config.get("sampler_type", "euler"),
            ddim_eta=difflar_config.get("ddim_eta", 0.0),
        )

        # 基础配置参数
        self.max_latent_length = difflar_config.get("max_latent_length", 256)
        self.max_condition_length = difflar_config.get("max_condition_length", 128)
        self.diffusion_loss_weight = difflar_config.get("diffusion_loss_weight", 1.0)
        self.answer_loss_weight = difflar_config.get("answer_loss_weight", 5.0)
        self.alignment_loss_weight = difflar_config.get("alignment_loss_weight", 1.0)
        self.num_inference_steps = difflar_config.get("num_inference_steps", 32)  # 关键：从 128 减少到 32
        self.train_inference_steps = difflar_config.get("train_inference_steps", 10)
        self.use_cfg = difflar_config.get("use_cfg", False)
        self.cfg_scale = difflar_config.get("cfg_scale", 1.5)
        self.clamp_value = difflar_config.get("clamp_value", 3.0)
        
        # ═══ 两阶段训练配置 ═══
        self.stage1_epochs = difflar_config.get("stage1_epochs", 10)
        self.training_stage = 1  # 当前训练阶段
        
        # ═══ Stage1a/Stage1b 配置 ═══
        self.stage1a_ratio = difflar_config.get("stage1a_ratio", 0.5)
        self.stage1a_train_inference_steps = difflar_config.get("stage1a_train_inference_steps", 10)
        self.stage1b_train_inference_steps = difflar_config.get("stage1b_train_inference_steps", 20)
        self.stage1b_high_noise_ratio = difflar_config.get("stage1b_high_noise_ratio", 0.5)
        
        # ═══ Self-Conditioning配置 ═══
        self.stage1_self_cond_prob = difflar_config.get("stage1_self_cond_prob", 0.5)
        self.stage2_self_cond_prob = difflar_config.get("stage2_self_cond_prob", 1.0)
        
        # Stage1: 冻结 LLM，只训练 Diffusion
        # Stage2: 解冻 LLM（LoRA），联合训练
        self._freeze_llm()

        # Stage1 完全解耦：可选不加载/不注册 LLM 权重，避免显存占用
        # 用法：Stage1 运行时传参 difflar_config.stage1_no_llm=true
        self.stage1_no_llm = bool(difflar_config.get("stage1_no_llm", False))
        if self.training_stage == 1 and self.stage1_no_llm:
            self._drop_llm_for_stage1()
        
        # Loss记录
        self.loss_history = []
        self.loss_csv_path = None

    def _freeze_llm(self):
        """Stage1 时冻结 LLM 参数，只训练 Diffusion Model"""
        if getattr(self, "llm", None) is None:
            return
        for param in self.llm.parameters():
            param.requires_grad = False
        if getattr(self, "embedding", None) is not None:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def _unfreeze_llm(self):
        """Stage2 时解冻 LLM 参数（LoRA）"""
        if getattr(self, "llm", None) is None:
            raise RuntimeError("LLM is not loaded. Run Stage2 with difflar_config.stage1_no_llm=false.")
        # LoRA 参数默认是可训练的，这里确保它们被启用
        for name, param in self.llm.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

    def _drop_llm_for_stage1(self):
        """
        Stage1 只训练 Diffusion：删除 LLM/Embedding 模块，避免 Lightning 将其搬到 GPU。
        注意：这会禁用 Stage2/推理相关功能；Stage2 请用 stage1_no_llm=false 重新启动训练。
        """
        try:
            del self.llm
        except Exception:
            pass
        try:
            del self.embedding
        except Exception:
            pass
        self.llm = None
        self.embedding = None

    def _pad_to_fixed_length(
        self, 
        hidden_states: torch.Tensor, 
        mask: torch.Tensor, 
        max_length: int
    ) -> tuple:
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
            # Padding：使用零向量
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

    def _forward_stage1(self, batch):
        """
        Stage1 前向传播：只训练 Diffusion 模型
        
        Args:
            batch: 包含预保存的 Hidden States
                - question_embeds: [B, L_q, H]  # Question Embeddings（条件输入）
                - steps_hidden: [B, L_s, H]      # Steps Hidden State（学习目标）
                - question_mask: [B, L_q]
                - steps_mask: [B, L_s]
        """
        # 1. 从 batch 中获取预保存的 Hidden States
        # 注意：HiddenStateDataset 可能以 float16 落盘以节省空间；这里统一转 float32 保证训练数值稳定
        question_embeds = batch["question_embeds"].float()  # [B, L_q, H]
        steps_hidden = batch["steps_hidden"].float()  # [B, L_s, H]
        question_mask = batch["question_mask"].float()  # [B, L_q]
        steps_mask = batch["steps_mask"].float()  # [B, L_s]
        
        # 2. 将变长的 Hidden States 转换为固定长度
        question_cond_padded, question_mask_padded = self._pad_to_fixed_length(
            question_embeds, question_mask, self.max_condition_length
        )
        steps_hidden_padded, steps_mask_padded = self._pad_to_fixed_length(
            steps_hidden, steps_mask, self.max_latent_length
        )
        
        # 3. 判断当前是 Stage1a 还是 Stage1b
        stage1a_epochs = int(self.stage1_epochs * self.stage1a_ratio)
        is_stage1b = self.current_epoch >= stage1a_epochs
        
        # 4. 计算 Diffusion Loss
        # Stage1a：正常时间步采样（t ~ U[0, 1]）
        # Stage1b：强化高噪声采样（50% t ~ U[0.8, 1.0]，50% t ~ U[0, 1]）
        
        # 设置 Self-Conditioning 概率
        self.latent_diffusion.self_cond_prob = self.stage1_self_cond_prob
        
        # Stage1b 高噪声采样：修改 Flow Matching 的时间采样范围
        if is_stage1b and random.random() < self.stage1b_high_noise_ratio:
            # 临时修改 Flow Matching 的时间采样范围
            # 注意：这需要在 LatentDiffusion 中添加支持，当前先使用正常采样
            # 高噪声训练的效果主要通过增加 train_inference_steps 来实现
            pass
        
        # 计算 Diffusion Loss
        diffusion_loss, _, _ = self.latent_diffusion(
            steps_embeds=steps_hidden_padded,
            condition=question_cond_padded,
            attention_mask=steps_mask_padded,
            condition_mask=question_mask_padded,
            use_self_cond=None,  # 按概率决定
        )
        
        # 5. 生成 Steps Hidden State（用于 Alignment Loss）
        train_inference_steps = self.stage1b_train_inference_steps if is_stage1b else self.stage1a_train_inference_steps
        generated_steps_hidden = self.latent_diffusion.generate(
            condition=question_cond_padded,
            num_inference_steps=train_inference_steps,
            latent_length=self.max_latent_length,
            condition_mask=question_mask_padded,
            enable_grad=True,  # 保留梯度
            use_self_cond=True,
        )  # [B, L_s, H]
        
        # 6. Alignment Loss（显式对齐生成的 Hidden State 和 GT）
        alignment_loss = self._compute_alignment_loss(
            generated_steps_hidden, 
            steps_hidden_padded, 
            steps_mask_padded
        )
        
        # 7. Stage1 总损失：Diffusion Loss + Alignment Loss
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
        self.latent_diffusion.self_cond_prob = self.stage2_self_cond_prob
        diffusion_loss, _, _ = self.latent_diffusion(
            steps_embeds=gt_steps_hidden_padded,
            condition=question_hidden_padded,
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
        question_hidden_length = question_hidden.shape[1]
        all_embeds = torch.cat([
            question_hidden,  # Question Hidden State
            generated_steps_hidden,  # Steps Hidden State
            answer_embeds  # Answer Embedding
        ], dim=1)
        
        all_attention_mask = torch.cat([
            torch.ones(batch_size, question_hidden_length, device=self.device, dtype=question_attention_mask.dtype),
            torch.ones(batch_size, self.max_latent_length, device=self.device, dtype=question_attention_mask.dtype),
            answer_attention_mask
        ], dim=1)
        
        # 创建 labels（只计算 answer 部分的 loss）
        labels = torch.cat([
            torch.full((batch_size, question_hidden_length + self.max_latent_length), -100, device=self.device),
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

    def forward(self, batch):
        """
        根据训练阶段选择前向传播方法
        """
        if self.training_stage == 1:
            return self._forward_stage1(batch)
        else:
            return self._forward_stage2(batch)

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

    @torch.no_grad()
    def latent_generate(
        self,
        questions: List[str],
        return_latent_hidden_states: bool = False,
    ):
        """
        推理时生成答案
        
        Args:
            questions: 问题列表
            return_latent_hidden_states: 是否返回 Steps Hidden State
        
        Returns:
            pred_ids: 生成的答案 token ids
            n_latent_forward: Steps 长度（固定为 max_latent_length）
            steps_hidden: Steps Hidden State（可选）
        """
        answer_generation_config = self.model_kwargs.answer_generation_config
        batch_size = len(questions)
        
        # 1. 编码 Question
        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        question_embeds = self.embedding(question_input_ids)
        question_mask = question_attention_mask.float()
        
        # 2. 获取 Question Hidden State
        question_outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=question_attention_mask,
            output_hidden_states=True,
        )
        question_hidden = question_outputs.hidden_states[-1]  # [B, L_q, H]
        
        # 3. 将 Question Hidden State 转换为固定长度
        question_hidden_padded, question_mask_padded = self._pad_to_fixed_length(
            question_hidden, question_mask, self.max_condition_length
        )
        
        # 4. Diffusion 生成 Steps Hidden State（32步）
        if self.use_cfg:
            steps_hidden = self.latent_diffusion.generate_with_cfg(
                condition=question_hidden_padded,
                num_inference_steps=self.num_inference_steps,
                latent_length=self.max_latent_length,
                cfg_scale=self.cfg_scale,
                condition_mask=question_mask_padded,
                clamp_value=self.clamp_value,
                use_self_cond=True,
            )
        else:
            steps_hidden = self.latent_diffusion.generate(
                condition=question_hidden_padded,
                num_inference_steps=self.num_inference_steps,
                latent_length=self.max_latent_length,
                condition_mask=question_mask_padded,
                clamp_value=self.clamp_value,
                use_self_cond=True,
            )
        
        # 5. 准备 Separator
        sep_text = [self.thinking_separator] * batch_size
        sep_inputs = self.tokenizer(
            sep_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        sep_embeds = self.embedding(sep_inputs.input_ids)
        sep_length = sep_embeds.shape[1]
        
        # 6. 拼接并生成 Answer
        question_hidden_length = question_hidden.shape[1]
        all_embeds = torch.cat([
            question_hidden,  # Question Hidden State
            steps_hidden,     # Steps Hidden State
            sep_embeds        # Separator Embedding
        ], dim=1)
        
        all_attention_mask = torch.cat([
            torch.ones(batch_size, question_hidden_length, device=self.device, dtype=question_attention_mask.dtype),
            torch.ones(batch_size, self.max_latent_length, device=self.device, dtype=question_attention_mask.dtype),
            torch.ones(batch_size, sep_length, device=self.device, dtype=question_attention_mask.dtype),
        ], dim=1)
        
        pred_ids = self.llm.generate(
            inputs_embeds=all_embeds,
            attention_mask=all_attention_mask,
            **answer_generation_config,
        )
        
        n_latent_forward = torch.full(
            (batch_size, 1), self.max_latent_length, device=self.device, dtype=torch.long
        )
        
        if return_latent_hidden_states:
            return pred_ids, n_latent_forward, steps_hidden
        return pred_ids, n_latent_forward

    def on_train_epoch_start(self):
        """每个 epoch 开始时检查是否切换阶段"""
        super().on_train_epoch_start() if hasattr(super(), 'on_train_epoch_start') else None
        
        # 检查是否需要切换到 Stage 2
        if self.current_epoch >= self.stage1_epochs and self.training_stage == 1:
            self.training_stage = 2
            self._unfreeze_llm()  # 解冻 LLM（LoRA）
            print(f"\n{'='*60}")
            print(f"Switching to Stage 2: Joint Training")
            print(f"  - Self-Cond probability: {self.stage2_self_cond_prob:.1%}")
            print(f"{'='*60}\n")

    def on_fit_start(self):
        """训练开始时初始化"""
        super().on_fit_start() if hasattr(super(), 'on_fit_start') else None
        
        # 初始化CSV
        if self.trainer.global_rank == 0:
            log_dir = self.trainer.logger.log_dir if self.trainer.logger else "logs/difflar_hidden"
            os.makedirs(log_dir, exist_ok=True)
            self.loss_csv_path = os.path.join(log_dir, "loss_history.csv")
            with open(self.loss_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'epoch', 'stage', 'stage1_substage',
                               'total_loss', 'diffusion_loss', 'alignment_loss', 'answer_loss'])
        
        # 估计归一化参数（如果 Stage1 使用 Hidden State Dataset）
        if self.training_stage == 1 and hasattr(self.trainer, 'datamodule'):
            if self.latent_diffusion.normalize_latent and not self.latent_diffusion._latent_stats_initialized.item():
                self._estimate_hidden_state_stats(self.trainer.datamodule)
        
        # 打印训练配置
        if self.trainer.global_rank == 0:
            print(f"\n{'='*60}")
            print("DiffLaR Hidden Training Configuration")
            print(f"{'='*60}")
            print(f"Stage 1 epochs: {self.stage1_epochs}")
            print(f"Stage 1a ratio: {self.stage1a_ratio:.1%}")
            print(f"Stage 1 Self-Cond prob: {self.stage1_self_cond_prob:.1%}")
            print(f"Stage 2 Self-Cond prob: {self.stage2_self_cond_prob:.1%}")
            print(f"Inference steps: {self.num_inference_steps}")
            print(f"{'='*60}\n")

    def _estimate_hidden_state_stats(self, datamodule=None):
        """从训练数据估计 Hidden State 的归一化参数"""
        print("Estimating hidden state statistics from training data...")
        
        if datamodule is not None:
            train_loader = datamodule.train_dataloader()
        elif hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            train_loader = self.trainer.datamodule.train_dataloader()
        else:
            print("Warning: Cannot get train_dataloader, skipping hidden state stats estimation")
            return
        
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
        
        mean, std, scale = self.latent_diffusion.estimate_latent_stats(all_hidden, all_masks)
        
        print(f"Hidden state stats estimated: Mean norm={mean.norm():.4f}, "
              f"Std mean={std.mean():.6f}, Scale={scale:.4f}")

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        """训练步骤"""
        log_dict = self.forward(batch=batch)
        
        # 记录Loss
        if self.trainer.global_rank == 0:
            step = self.global_step
            loss_record = {
                'step': step,
                'epoch': self.current_epoch,
                'stage': self.training_stage,
                'stage1_substage': log_dict.get('stage1_substage', 0.0),
                'total_loss': log_dict['total_loss'].item(),
                'diffusion_loss': log_dict['diffusion_loss'].item(),
                'alignment_loss': log_dict.get('alignment_loss', torch.tensor(0.0)).item() if torch.is_tensor(log_dict.get('alignment_loss', 0.0)) else log_dict.get('alignment_loss', 0.0),
                'answer_loss': log_dict.get('answer_loss', torch.tensor(0.0)).item() if torch.is_tensor(log_dict.get('answer_loss', 0.0)) else log_dict.get('answer_loss', 0.0),
            }
            self.loss_history.append(loss_record)
            
            if self.loss_csv_path:
                with open(self.loss_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        step, loss_record['epoch'], loss_record['stage'],
                        loss_record['stage1_substage'], loss_record['total_loss'], 
                        loss_record['diffusion_loss'], loss_record['alignment_loss'],
                        loss_record['answer_loss']
                    ])
        
        # 添加前缀并记录
        log_dict_filtered = {f"train/{k}": v for k, v in log_dict.items() 
                            if k not in ['training_stage', 'stage1_substage']}
        log_dict_filtered['train/stage'] = float(self.training_stage)
        if 'stage1_substage' in log_dict:
            log_dict_filtered['train/stage1_substage'] = log_dict['stage1_substage']
        
        self.log_dict(log_dict_filtered, sync_dist=True, prog_bar=True, 
                     batch_size=self.all_config.dataloader.batch_size)
        return log_dict["total_loss"]

    def on_fit_end(self):
        """训练结束时绘制Loss曲线"""
        super().on_fit_end() if hasattr(super(), 'on_fit_end') else None
        if self.trainer.global_rank == 0 and len(self.loss_history) > 0:
            self._plot_loss_curves()

    def _plot_loss_curves(self):
        """绘制并保存Loss曲线图"""
        steps = [r['step'] for r in self.loss_history]
        total_loss = [r['total_loss'] for r in self.loss_history]
        diffusion_loss = [r['diffusion_loss'] for r in self.loss_history]
        alignment_loss = [r['alignment_loss'] for r in self.loss_history]
        answer_loss = [r['answer_loss'] for r in self.loss_history]
        stages = [r['stage'] for r in self.loss_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 找到阶段切换点
        stage_switch = None
        for i, s in enumerate(stages):
            if s == 2:
                stage_switch = steps[i]
                break
        
        # Total Loss
        axes[0, 0].plot(steps, total_loss, 'b-', alpha=0.7)
        if stage_switch:
            axes[0, 0].axvline(x=stage_switch, color='r', linestyle='--', label='Stage 2 Start')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Diffusion Loss
        axes[0, 1].plot(steps, diffusion_loss, 'g-', alpha=0.7)
        if stage_switch:
            axes[0, 1].axvline(x=stage_switch, color='r', linestyle='--', label='Stage 2 Start')
        axes[0, 1].set_title('Diffusion Loss')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Alignment Loss
        axes[1, 0].plot(steps, alignment_loss, 'orange', alpha=0.7, label='Alignment')
        if stage_switch:
            axes[1, 0].axvline(x=stage_switch, color='r', linestyle='--', label='Stage 2 Start')
        axes[1, 0].set_title('Alignment Loss')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # All Losses
        axes[1, 1].plot(steps, total_loss, 'b-', label='Total', alpha=0.7)
        axes[1, 1].plot(steps, diffusion_loss, 'g-', label='Diffusion', alpha=0.7)
        axes[1, 1].plot(steps, alignment_loss, 'orange', label='Alignment', alpha=0.7)
        axes[1, 1].plot(steps, answer_loss, 'r-', label='Answer', alpha=0.7)
        if stage_switch:
            axes[1, 1].axvline(x=stage_switch, color='k', linestyle='--', label='Stage 2 Start')
        axes[1, 1].set_title('All Losses (DiffLaR Hidden)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        log_dir = self.trainer.logger.log_dir if self.trainer.logger else "logs/difflar_hidden"
        save_path = os.path.join(log_dir, "loss_curves_hidden.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss curves saved to: {save_path}")

