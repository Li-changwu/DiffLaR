import os
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Optional

from .model_base import LitCoTModelBase
from ..modules.diffusion import LatentDiffusion
from ..utils.utils import get_position_ids_from_attention_mask


class LitDiffLaRFused(LitCoTModelBase):
    """
    DiffLaR Fused: 融合两阶段训练 + Self-Conditioning
    
    核心改进：
    1. Stage 1: Teacher-Forcing + Self-Conditioning基础学习
       - Diffusion Loss用GT latent
       - 50%概率启用Self-Conditioning
       
    2. Stage 2: Rollout Training + Self-Conditioning强化
       - Diffusion Loss用自生成latent（Rollout）
       - 100%启用Self-Conditioning
       - Curriculum: 逐步增加Rollout比例
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

        # Latent Diffusion模型
        self.latent_diffusion = LatentDiffusion(
            hidden_size=self.llm.config.hidden_size,
            num_timesteps=difflar_config.get("num_timesteps", 1000),
            num_layers=difflar_config.get("denoiser_layers", 6),
            num_heads=difflar_config.get("denoiser_heads", 8),
            max_latent_length=difflar_config.get("max_latent_length", 256),
            schedule_type=difflar_config.get("noise_schedule", "linear"),
            dropout=difflar_config.get("dropout", 0.1),
            normalize_latent=difflar_config.get("normalize_latent", True),
            prediction_type=difflar_config.get("prediction_type", "epsilon"),
            sampler_type=difflar_config.get("sampler_type", "ddim"),
            ddim_eta=difflar_config.get("ddim_eta", 0.0),
        )

        # 基础配置参数
        self.max_latent_length = difflar_config.get("max_latent_length", 256)
        self.diffusion_loss_weight = difflar_config.get("diffusion_loss_weight", 1.0)
        self.answer_loss_weight = difflar_config.get("answer_loss_weight", 5.0)
        self.alignment_loss_weight = difflar_config.get("alignment_loss_weight", 1.0)  # 中间监督权重
        self.num_inference_steps = difflar_config.get("num_inference_steps", 128)
        self.train_inference_steps = difflar_config.get("train_inference_steps", 10)
        self.use_cfg = difflar_config.get("use_cfg", False)
        self.cfg_scale = difflar_config.get("cfg_scale", 1.5)
        self.clamp_value = difflar_config.get("clamp_value", 3.0)
        
        # ═══ 两阶段训练配置 ═══
        self.stage1_epochs = difflar_config.get("stage1_epochs", 5)
        self.training_stage = 1  # 当前训练阶段
        
        # ═══ Rollout训练配置 ═══
        self.rollout_start_ratio = difflar_config.get("rollout_start_ratio", 0.3)
        self.rollout_final_ratio = difflar_config.get("rollout_final_ratio", 1.0)
        self.rollout_inference_steps = difflar_config.get("rollout_inference_steps", 10)
        
        # ═══ Self-Conditioning配置 ═══
        self.stage1_self_cond_prob = difflar_config.get("stage1_self_cond_prob", 0.5)
        self.stage2_self_cond_prob = difflar_config.get("stage2_self_cond_prob", 1.0)
        
        # 冻结LLM参数，只训练Diffusion Model
        self._freeze_llm()
        
        # Loss记录
        self.loss_history = []
        self.loss_csv_path = None
        
        # 注意：归一化状态现在由 latent_diffusion._latent_stats_initialized buffer 管理

    def _freeze_llm(self):
        """冻结LLM参数，只训练Diffusion Model"""
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.embedding.parameters():
            param.requires_grad = False

    def prepare_fixed_length_steps(
        self,
        steps: List[str],
        max_length: Optional[int] = None,
    ) -> tuple:
        """将变长的Steps文本转换为固定长度的Embedding"""
        if max_length is None:
            max_length = self.max_latent_length

        batch_size = len(steps)

        steps_text = [self.steps_template.format(s) for s in steps]
        inputs = self.tokenizer.batch_encode_plus(
            steps_text,
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        steps_input_ids = inputs["input_ids"].to(self.device)
        steps_attention_mask = inputs["attention_mask"].to(self.device)

        steps_embeds = self.embedding(steps_input_ids)
        current_length = steps_embeds.shape[1]

        if current_length > max_length:
            steps_embeds = steps_embeds[:, :max_length, :]
            steps_mask = steps_attention_mask[:, :max_length]
        elif current_length < max_length:
            pad_length = max_length - current_length
            # 使用 PAD token 的 embedding 而非 0 向量
            # 0 向量在 embedding 空间中不是"无意义"，而是一个具体的点
            pad_token_embed = self.embedding(
                torch.tensor([[self.tokenizer.pad_token_id]], device=self.device)
            )  # [1, 1, H]
            pad_embeds = pad_token_embed.expand(batch_size, pad_length, -1)  # [B, pad_length, H]
            steps_embeds = torch.cat([steps_embeds, pad_embeds], dim=1)
            steps_mask = F.pad(steps_attention_mask, (0, pad_length), value=0)
        else:
            steps_mask = steps_attention_mask

        steps_mask = steps_mask.float()
        return steps_embeds, steps_mask

    def get_current_rollout_ratio(self) -> float:
        """Curriculum: 计算当前Rollout比例"""
        if self.training_stage == 1:
            return 0.0
        
        stage2_epoch = self.current_epoch - self.stage1_epochs
        total_stage2_epochs = self.trainer.max_epochs - self.stage1_epochs
        
        if total_stage2_epochs <= 0:
            return self.rollout_final_ratio
        
        progress = min(1.0, stage2_epoch / max(total_stage2_epochs, 1))
        ratio = self.rollout_start_ratio + (self.rollout_final_ratio - self.rollout_start_ratio) * progress
        return ratio

    def get_current_self_cond_prob(self) -> float:
        """获取当前Self-Conditioning概率"""
        if self.training_stage == 1:
            return self.stage1_self_cond_prob
        return self.stage2_self_cond_prob

    def forward(self, batch):
        """
        融合训练的前向传播
        
        Stage 1: Teacher-Forcing + Self-Conditioning
        Stage 2: Rollout + Self-Conditioning
        """
        difflar_config = self.model_kwargs.difflar_config

        # 0. 准备输入
        question = batch["question"]
        steps = batch["steps"]
        answer = batch["answer"]
        batch_size = len(question)

        # 1. Question -> Embedding
        question_input_ids, question_attention_mask = self.prepare_inputs(
            question,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        query_embedding = self.embedding(question_input_ids)
        query_mask = question_attention_mask.float()

        # 2. 准备GT Steps Embedding
        gt_steps_embeds, steps_mask = self.prepare_fixed_length_steps(steps)

        # 3. 根据训练阶段计算Diffusion Loss
        if self.training_stage == 1:
            # ═══ Stage 1: Teacher-Forcing + Self-Conditioning ═══
            # 设置Self-Conditioning概率
            self.latent_diffusion.self_cond_prob = self.stage1_self_cond_prob
            
            # Diffusion Loss用GT
            diffusion_loss, _, _ = self.latent_diffusion(
                steps_embeds=gt_steps_embeds,
                condition=query_embedding,
                attention_mask=steps_mask,
                condition_mask=query_mask,
            )
            
            # [改进] Stage 1 也使用生成的 latent + Alignment Loss
            # 让 LLM 从一开始就适应生成的 embedding，避免 Exposure Bias
            generated_steps_embeds = self._generate_for_training(query_embedding, query_mask)
            
            # 中间监督：Latent Alignment Loss（对齐生成和GT）
            alignment_loss = self._compute_alignment_loss(
                generated_steps_embeds, gt_steps_embeds, steps_mask
            )
            
        else:
            # ═══ Stage 2: Rollout + Self-Conditioning ═══
            rollout_ratio = self.get_current_rollout_ratio()
            self.latent_diffusion.self_cond_prob = self.stage2_self_cond_prob
            
            # [修复] Diffusion Loss必须永远指向GT，不能用生成的latent作为目标！
            # Rollout的目的是让模型在"自身产生误差"时依然能修正方向指向GT
            diffusion_loss, _, _ = self.latent_diffusion(
                steps_embeds=gt_steps_embeds,  # 永远用GT作为学习目标
                condition=query_embedding,
                attention_mask=steps_mask,
                condition_mask=query_mask,
                use_self_cond=True,
            )
            
            # Answer Loss使用生成的latent（这才是Rollout训练的核心）
            if random.random() < rollout_ratio:
                # Rollout: 用生成的latent计算Answer Loss
                generated_steps_embeds = self._generate_for_training(query_embedding, query_mask)
                # 中间监督：Latent Alignment Loss
                alignment_loss = self._compute_alignment_loss(
                    generated_steps_embeds, gt_steps_embeds, steps_mask
                )
            else:
                # 保持部分GT训练，稳定训练
                generated_steps_embeds = gt_steps_embeds
                alignment_loss = torch.tensor(0.0, device=self.device)

        # 4. Answer Loss
        # Stage 1: 使用真实的 steps_mask
        # Stage 2: 引入 Mask Dropout，模拟推理时的"全关注"状态
        if self.training_stage == 1:
            answer_steps_mask = steps_mask
        else:
            # Mask Dropout: 50% 概率使用全 1 mask
            # 让 LLM 学会：即使 mask=1，看到 PAD embedding 模式时也要忽略
            # 配合 PAD embedding 使用，确保训练-推理一致性
            if random.random() < 0.5:
                answer_steps_mask = torch.ones(batch_size, self.max_latent_length, device=self.device)
            else:
                answer_steps_mask = steps_mask
        
        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answer,
            padding_side="right",
            part="answer",
            prefix=self.thinking_separator,
            suffix=self.tokenizer.eos_token,
        )
        answer_embeds = self.embedding(answer_input_ids)

        # 拼接: Question + Steps + Answer
        all_embeds = torch.cat([query_embedding, generated_steps_embeds, answer_embeds], dim=1)
        all_attention_mask = torch.cat(
            [question_attention_mask, answer_steps_mask.long(), answer_attention_mask], dim=1
        )

        question_length = query_embedding.shape[1]
        steps_length = generated_steps_embeds.shape[1]

        # 创建labels（只计算answer部分的loss）
        labels = torch.cat(
            [
                torch.full((batch_size, question_length + steps_length), -100, device=self.device),
                answer_input_ids,
            ],
            dim=1,
        )
        labels[labels == self.tokenizer.pad_token_id] = -100

        position_ids = get_position_ids_from_attention_mask(all_attention_mask)

        answer_outputs = self.llm.forward(
            inputs_embeds=all_embeds,
            attention_mask=all_attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
        answer_loss = answer_outputs.loss

        # 5. 总损失构造（分阶段策略 + 中间监督）
        if self.training_stage == 1:
            # Stage 1: Diffusion + Alignment + Answer（权重较低）
            # 中间监督帮助 latent 更快对齐 GT，减少 Exposure Bias
            total_loss = (
                self.diffusion_loss_weight * diffusion_loss
                + self.alignment_loss_weight * alignment_loss
                + 0.5 * answer_loss  # Stage 1 answer loss 权重较低
            )
            answer_loss_for_log = answer_loss
        else:
            # Stage 2: 联合训练（全权重）
            total_loss = (
                self.diffusion_loss_weight * diffusion_loss
                + self.alignment_loss_weight * alignment_loss
                + self.answer_loss_weight * answer_loss
            )
            answer_loss_for_log = answer_loss

        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "alignment_loss": alignment_loss.detach() if torch.is_tensor(alignment_loss) else alignment_loss,
            "answer_loss": answer_loss_for_log,
            "training_stage": float(self.training_stage),
            "rollout_ratio": self.get_current_rollout_ratio(),
        }

    def _compute_alignment_loss(
        self,
        generated_embeds: torch.Tensor,
        gt_embeds: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算中间监督的对齐损失
        
        三种对齐方式的组合：
        1. MSE Loss: 逐元素对齐，保证数值接近
        2. Cosine Loss: 方向对齐，保证语义方向一致
        3. Feature-wise Loss: 每个位置的特征向量对齐
        
        Args:
            generated_embeds: 生成的 latent [B, L, H]
            gt_embeds: GT latent [B, L, H]
            mask: 有效位置掩码 [B, L]
            
        Returns:
            alignment_loss: 标量损失
        """
        # 扩展 mask 到 hidden 维度
        mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
        
        # 1. MSE Loss（只在有效位置计算）
        mse_loss = F.mse_loss(
            generated_embeds * mask_expanded,
            gt_embeds * mask_expanded,
            reduction='sum'
        ) / (mask.sum() * generated_embeds.shape[-1] + 1e-8)
        
        # 2. Cosine Similarity Loss（方向对齐）
        # 对每个位置的 hidden vector 计算 cosine similarity
        gen_norm = F.normalize(generated_embeds, p=2, dim=-1)
        gt_norm = F.normalize(gt_embeds, p=2, dim=-1)
        cosine_sim = (gen_norm * gt_norm).sum(dim=-1)  # [B, L]
        cosine_loss = (1 - cosine_sim) * mask  # 只在有效位置
        cosine_loss = cosine_loss.sum() / (mask.sum() + 1e-8)
        
        # 组合损失（MSE 为主，Cosine 为辅）
        total_alignment_loss = mse_loss + 0.1 * cosine_loss
        
        return total_alignment_loss

    def _generate_for_training(self, query_embedding, query_mask):
        """
        生成用于训练的latent（保留梯度）
        
        注意：训练时强制不使用 CFG 以节省显存
        CFG 需要每步 2 次 forward，10 步 Rollout = 20 次计算图，极易 OOM
        训练目的是让模型本体学会修正，而非依赖 CFG 技巧
        """
        return self.latent_diffusion.generate(
            condition=query_embedding,
            num_inference_steps=self.rollout_inference_steps if self.training_stage == 2 else self.train_inference_steps,
            latent_length=self.max_latent_length,
            condition_mask=query_mask,
            enable_grad=True,
            clamp_value=self.clamp_value,
            use_self_cond=True,
        )

    @torch.no_grad()
    def latent_generate(
        self,
        questions: List[str],
        return_latent_hidden_states: bool = False,
    ):
        """推理时生成答案"""
        answer_generation_config = self.model_kwargs.answer_generation_config
        batch_size = len(questions)

        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        query_embedding = self.embedding(question_input_ids)
        query_mask = question_attention_mask.float()

        # Diffusion生成Steps Embedding（启用Self-Conditioning）
        if self.use_cfg:
            steps_embeds = self.latent_diffusion.generate_with_cfg(
                condition=query_embedding,
                num_inference_steps=self.num_inference_steps,
                latent_length=self.max_latent_length,
                cfg_scale=self.cfg_scale,
                condition_mask=query_mask,
                clamp_value=self.clamp_value,
                use_self_cond=True,
            )
        else:
            steps_embeds = self.latent_diffusion.generate(
                condition=query_embedding,
                num_inference_steps=self.num_inference_steps,
                latent_length=self.max_latent_length,
                condition_mask=query_mask,
                clamp_value=self.clamp_value,
                use_self_cond=True,
            )

        # 拼接并生成答案
        # 统一使用 Tokenizer 处理 separator，确保与训练时一致
        sep_text = [self.thinking_separator] * batch_size
        sep_inputs = self.tokenizer(
            sep_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        sep_embeds = self.embedding(sep_inputs.input_ids)
        sep_length = sep_embeds.shape[1]

        all_embeds = torch.cat([query_embedding, steps_embeds, sep_embeds], dim=1)
        all_attention_mask = torch.cat(
            [
                question_attention_mask,
                torch.ones(batch_size, self.max_latent_length, device=self.device, dtype=question_attention_mask.dtype),
                torch.ones(batch_size, sep_length, device=self.device, dtype=question_attention_mask.dtype),
            ],
            dim=1,
        )

        pred_ids = self.llm.generate(
            inputs_embeds=all_embeds,
            attention_mask=all_attention_mask,
            **answer_generation_config,
        )

        n_latent_forward = torch.full(
            (batch_size, 1), self.max_latent_length, device=self.device, dtype=torch.long
        )

        if return_latent_hidden_states:
            return pred_ids, n_latent_forward, steps_embeds
        return pred_ids, n_latent_forward

    def on_train_epoch_start(self):
        """每个epoch开始时检查是否切换阶段"""
        super().on_train_epoch_start() if hasattr(super(), 'on_train_epoch_start') else None
        
        # 检查是否需要切换到Stage 2
        if self.current_epoch >= self.stage1_epochs and self.training_stage == 1:
            self.training_stage = 2
            print(f"\n{'='*60}")
            print(f"Switching to Stage 2: Rollout Training")
            print(f"  - Rollout ratio starts at: {self.rollout_start_ratio:.1%}")
            print(f"  - Self-Cond probability: {self.stage2_self_cond_prob:.1%}")
            print(f"{'='*60}\n")

    def on_fit_start(self):
        """训练开始时初始化"""
        super().on_fit_start() if hasattr(super(), 'on_fit_start') else None
        
        # 初始化CSV
        if self.trainer.global_rank == 0:
            log_dir = self.trainer.logger.log_dir if self.trainer.logger else "logs/difflar_fused"
            os.makedirs(log_dir, exist_ok=True)
            self.loss_csv_path = os.path.join(log_dir, "loss_history.csv")
            with open(self.loss_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'epoch', 'stage', 'rollout_ratio', 
                               'total_loss', 'diffusion_loss', 'alignment_loss', 'answer_loss'])
        
        # 估计归一化参数
        if self.latent_diffusion.normalize_latent and not self.latent_diffusion._latent_stats_initialized.item():
            self._estimate_latent_stats(self.trainer.datamodule)
        
        # 打印训练配置
        if self.trainer.global_rank == 0:
            print(f"\n{'='*60}")
            print("DiffLaR Fused Training Configuration")
            print(f"{'='*60}")
            print(f"Stage 1 epochs: {self.stage1_epochs}")
            print(f"Stage 1 Self-Cond prob: {self.stage1_self_cond_prob:.1%}")
            print(f"Stage 2 Self-Cond prob: {self.stage2_self_cond_prob:.1%}")
            print(f"Rollout ratio: {self.rollout_start_ratio:.1%} → {self.rollout_final_ratio:.1%}")
            print(f"{'='*60}\n")

    def _estimate_latent_stats(self, datamodule=None):
        """从训练数据估计embedding的归一化参数"""
        print("Estimating latent statistics from training data...")
        
        if datamodule is not None:
            train_loader = datamodule.train_dataloader()
        elif hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            train_loader = self.trainer.datamodule.train_dataloader()
        else:
            print("Warning: Cannot get train_dataloader, skipping latent stats estimation")
            return
        
        all_embeds = []
        all_masks = []
        num_samples = 0
        max_samples = 500
        
        with torch.no_grad():
            for batch in train_loader:
                steps = batch["steps"]
                steps_embeds, steps_mask = self.prepare_fixed_length_steps(steps)
                all_embeds.append(steps_embeds.cpu())
                all_masks.append(steps_mask.cpu())
                num_samples += len(steps)
                if num_samples >= max_samples:
                    break
        
        all_embeds = torch.cat(all_embeds, dim=0).to(self.device)
        all_masks = torch.cat(all_masks, dim=0).to(self.device)
        
        mean, std, scale = self.latent_diffusion.estimate_latent_stats(all_embeds, all_masks)
        
        # 状态已在 latent_diffusion.set_latent_stats 中更新
        print(f"Latent stats estimated: Mean norm={mean.norm():.4f}, Std mean={std.mean():.6f}, Scale={scale:.4f}")

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
                'rollout_ratio': log_dict.get('rollout_ratio', 0.0),
                'total_loss': log_dict['total_loss'].item(),
                'diffusion_loss': log_dict['diffusion_loss'].item(),
                'alignment_loss': log_dict['alignment_loss'].item() if torch.is_tensor(log_dict['alignment_loss']) else log_dict['alignment_loss'],
                'answer_loss': log_dict['answer_loss'].item(),
            }
            self.loss_history.append(loss_record)
            
            if self.loss_csv_path:
                with open(self.loss_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, loss_record['epoch'], loss_record['stage'],
                                   loss_record['rollout_ratio'], loss_record['total_loss'], 
                                   loss_record['diffusion_loss'], loss_record['alignment_loss'],
                                   loss_record['answer_loss']])
        
        # 添加前缀并记录
        log_dict_filtered = {f"train/{k}": v for k, v in log_dict.items() 
                            if k not in ['training_stage', 'rollout_ratio']}
        log_dict_filtered['train/stage'] = float(self.training_stage)
        log_dict_filtered['train/rollout_ratio'] = self.get_current_rollout_ratio()
        
        self.log_dict(log_dict_filtered, sync_dist=True, prog_bar=True, 
                     batch_size=self.all_config.dataloader.batch_size)
        return log_dict["total_loss"]

    def on_test_start(self):
        """测试开始前确保latent stats已初始化"""
        super().on_test_start()
        
        # 如果latent stats未初始化，需要重新估计
        if self.latent_diffusion.normalize_latent and not self.latent_diffusion._latent_stats_initialized.item():
            print("Warning: Latent stats not initialized, re-estimating from test data...")
            self._estimate_latent_stats_from_test(self.trainer.datamodule)

    def _estimate_latent_stats_from_test(self, datamodule):
        """从测试数据估计latent stats（当没有训练数据时使用）"""
        print("Estimating latent statistics from test data...")
        
        # 确保datamodule已setup
        if not hasattr(datamodule, 'test_dataset') or datamodule.test_dataset is None:
            datamodule.setup('test')
        
        test_loader = datamodule.test_dataloader()
        
        all_embeds = []
        all_masks = []
        num_samples = 0
        max_samples = 500
        
        with torch.no_grad():
            for batch in test_loader:
                steps = batch["steps"]
                steps_embeds, steps_mask = self.prepare_fixed_length_steps(steps)
                all_embeds.append(steps_embeds.cpu())
                all_masks.append(steps_mask.cpu())
                num_samples += len(steps)
                if num_samples >= max_samples:
                    break
        
        all_embeds = torch.cat(all_embeds, dim=0).to(self.device)
        all_masks = torch.cat(all_masks, dim=0).to(self.device)
        
        mean, std, scale = self.latent_diffusion.estimate_latent_stats(all_embeds, all_masks)
        
        # 状态已在 latent_diffusion.set_latent_stats 中更新
        print(f"Latent stats estimated: Mean norm={mean.norm():.4f}, Std mean={std.mean():.6f}, Scale={scale:.4f}")

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
        
        # Answer Loss
        axes[1, 0].plot(steps, answer_loss, 'r-', alpha=0.7)
        if stage_switch:
            axes[1, 0].axvline(x=stage_switch, color='r', linestyle='--', label='Stage 2 Start')
        axes[1, 0].set_title('Answer Loss')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # All Losses
        axes[1, 1].plot(steps, total_loss, 'b-', label='Total', alpha=0.7)
        axes[1, 1].plot(steps, diffusion_loss, 'g-', label='Diffusion', alpha=0.7)
        axes[1, 1].plot(steps, answer_loss, 'r-', label='Answer', alpha=0.7)
        if stage_switch:
            axes[1, 1].axvline(x=stage_switch, color='k', linestyle='--', label='Stage 2 Start')
        axes[1, 1].set_title('All Losses (Fused Training)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        log_dir = self.trainer.logger.log_dir if self.trainer.logger else "logs/difflar_fused"
        save_path = os.path.join(log_dir, "loss_curves_fused.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss curves saved to: {save_path}")
