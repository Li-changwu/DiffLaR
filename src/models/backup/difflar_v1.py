import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Optional

from .model_base import LitCoTModelBase
from ..modules.diffusion import LatentDiffusion
from ..utils.utils import get_position_ids_from_attention_mask


class LitDiffLaR(LitCoTModelBase):
    """
    DiffLaR: Diffusion-based Latent Reasoning
    
    在LLM的Embedding空间使用Diffusion模型生成思考步骤
    
    训练流程:
        1. Query → LLM Embedding → Query Hidden
        2. Steps → LLM Embedding → Steps Embedding (固定长度)
        3. Diffusion Model学习: Query Hidden → Steps Embedding
        4. 两个Loss: Diffusion Loss + Answer Loss
    
    推理流程:
        1. Query → LLM → Query Hidden
        2. Diffusion生成 Steps Embedding
        3. Query + Steps Embedding → LLM → Answer
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
            normalize_latent=difflar_config.get("normalize_latent", True),  # 归一化
            prediction_type=difflar_config.get("prediction_type", "epsilon"),  # epsilon或x0
            sampler_type=difflar_config.get("sampler_type", "ddim"),  # ddpm或ddim
            ddim_eta=difflar_config.get("ddim_eta", 0.0),  # DDIM随机性
        )

        # 配置参数
        self.max_latent_length = difflar_config.get("max_latent_length", 256)
        self.diffusion_loss_weight = difflar_config.get("diffusion_loss_weight", 1.0)
        self.answer_loss_weight = difflar_config.get("answer_loss_weight", 5.0)
        self.num_inference_steps = difflar_config.get("num_inference_steps", 128)
        self.train_inference_steps = difflar_config.get("train_inference_steps", 10)
        self.use_cfg = difflar_config.get("use_cfg", False)
        self.cfg_scale = difflar_config.get("cfg_scale", 1.5)
        self.clamp_value = difflar_config.get("clamp_value", 3.0)  # 截断阈值
        
        # 冻结LLM参数，只训练Diffusion Model
        self._freeze_llm()
        
        # Loss记录
        self.loss_history = []
        self.loss_csv_path = None
        
        # 归一化状态标记
        self._latent_stats_estimated = False

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
        """
        将变长的Steps文本转换为固定长度的Embedding
        
        Args:
            steps: 思考步骤文本列表
            max_length: 最大长度，默认使用self.max_latent_length
            
        Returns:
            steps_embeds: 固定长度的Steps Embedding [B, max_length, H]
            steps_mask: 注意力掩码 [B, max_length]
        """
        if max_length is None:
            max_length = self.max_latent_length

        batch_size = len(steps)

        # 1. Tokenize steps
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

        # 2. 获取Embedding
        steps_embeds = self.embedding(steps_input_ids)  # [B, L, H]

        current_length = steps_embeds.shape[1]

        # 3. 截断或填充到固定长度
        if current_length > max_length:
            # 截断
            steps_embeds = steps_embeds[:, :max_length, :]
            steps_mask = steps_attention_mask[:, :max_length]
        elif current_length < max_length:
            # 右侧填充零向量
            pad_length = max_length - current_length
            steps_embeds = F.pad(steps_embeds, (0, 0, 0, pad_length), value=0)
            steps_mask = F.pad(steps_attention_mask, (0, pad_length), value=0)
        else:
            steps_mask = steps_attention_mask

        # 确保mask是float类型用于loss计算
        steps_mask = steps_mask.float()

        return steps_embeds, steps_mask

    def forward(self, batch):
        """
        训练时的前向传播
        
        计算两个Loss:
        1. Diffusion Loss: 预测噪声与真实噪声的MSE
        2. Answer Loss: 使用生成的Steps Embedding得到答案的CE Loss
        """
        difflar_config = self.model_kwargs.difflar_config

        # 0. 准备输入
        question = batch["question"]
        steps = batch["steps"]
        answer = batch["answer"]
        batch_size = len(question)

        # 1. Question -> Embedding (不经过LLM forward)
        question_input_ids, question_attention_mask = self.prepare_inputs(
            question,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        query_embedding = self.embedding(question_input_ids)  # [B, L_q, H]
        query_mask = question_attention_mask.float()  # [B, L_q]

        # 2. 准备固定长度的Steps Embedding
        steps_embeds, steps_mask = self.prepare_fixed_length_steps(steps)

        # 3. Diffusion Loss (Loss1)
        # 输入: Query Embedding作为条件, Steps Embedding作为目标
        diffusion_loss, _, _ = self.latent_diffusion(
            steps_embeds=steps_embeds,
            condition=query_embedding,
            attention_mask=steps_mask,
            condition_mask=query_mask,
        )

        # 4. Diffusion生成Steps Embedding（必须保留梯度以便回传到Diffusion Model）
        # 注意：enable_grad=True确保Answer Loss梯度可以回传到Diffusion
        if self.use_cfg:
            generated_steps_embeds = self.latent_diffusion.generate_with_cfg(
                condition=query_embedding,
                num_inference_steps=self.train_inference_steps,
                latent_length=self.max_latent_length,
                cfg_scale=self.cfg_scale,
                condition_mask=query_mask,
                enable_grad=True,
                clamp_value=self.clamp_value,
            )
        else:
            generated_steps_embeds = self.latent_diffusion.generate(
                condition=query_embedding,
                num_inference_steps=self.train_inference_steps,
                latent_length=self.max_latent_length,
                condition_mask=query_mask,
                enable_grad=True,
                clamp_value=self.clamp_value,
            )
        answer_steps_embeds = generated_steps_embeds
        answer_steps_mask = torch.ones(batch_size, self.max_latent_length, device=self.device)

        # 5. Answer Loss (Loss2)
        answer_input_ids, answer_attention_mask = self.prepare_inputs(
            answer,
            padding_side="right",
            part="answer",
            prefix=self.thinking_separator,
            suffix=self.tokenizer.eos_token,
        )
        answer_embeds = self.embedding(answer_input_ids)

        # 拼接: Question + Steps + Answer
        all_embeds = torch.cat([query_embedding, answer_steps_embeds, answer_embeds], dim=1)
        all_attention_mask = torch.cat(
            [question_attention_mask, answer_steps_mask.long(), answer_attention_mask], dim=1
        )

        question_length = query_embedding.shape[1]
        steps_length = answer_steps_embeds.shape[1]

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

        # 6. 总损失
        total_loss = (
            self.diffusion_loss_weight * diffusion_loss
            + self.answer_loss_weight * answer_loss
        )

        print("total_loss: ", total_loss.item())
        
        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "answer_loss": answer_loss,
        }

    @torch.no_grad()
    def compute_embedding_similarity(
        self,
        questions: List[str],
        steps: List[str],
    ) -> dict:
        """
        计算Diffusion生成的embedding与GT embedding的相似度
        用于验证Diffusion生成质量
        """
        # 1. Query Embedding
        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        query_embedding = self.embedding(question_input_ids)
        query_mask = question_attention_mask.float()

        # 2. GT Steps Embedding
        gt_steps_embeds, gt_steps_mask = self.prepare_fixed_length_steps(steps)

        # 3. Diffusion生成Steps Embedding（已包含反归一化）
        if self.use_cfg:
            gen_steps_embeds = self.latent_diffusion.generate_with_cfg(
                condition=query_embedding,
                num_inference_steps=self.num_inference_steps,
                latent_length=self.max_latent_length,
                cfg_scale=self.cfg_scale,
                condition_mask=query_mask,
                clamp_value=self.clamp_value,
            )
        else:
            gen_steps_embeds = self.latent_diffusion.generate(
                condition=query_embedding,
                num_inference_steps=self.num_inference_steps,
                latent_length=self.max_latent_length,
                condition_mask=query_mask,
                clamp_value=self.clamp_value,
            )

        # 4. 计算各种相似度指标
        # 只在有效位置计算
        valid_mask = gt_steps_mask.bool()  # [B, L]
        
        # 余弦相似度 (per position)
        cos_sim = F.cosine_similarity(gen_steps_embeds, gt_steps_embeds, dim=-1)  # [B, L]
        cos_sim_masked = (cos_sim * gt_steps_mask).sum() / (gt_steps_mask.sum() + 1e-8)
        
        # MSE
        mse = F.mse_loss(gen_steps_embeds, gt_steps_embeds, reduction='none').mean(dim=-1)  # [B, L]
        mse_masked = (mse * gt_steps_mask).sum() / (gt_steps_mask.sum() + 1e-8)
        
        # L2范数比较
        gen_norm = gen_steps_embeds.norm(dim=-1)  # [B, L]
        gt_norm = gt_steps_embeds.norm(dim=-1)  # [B, L]
        norm_ratio = gen_norm / (gt_norm + 1e-8)
        norm_ratio_masked = (norm_ratio * gt_steps_mask).sum() / (gt_steps_mask.sum() + 1e-8)
        
        return {
            "cos_sim": cos_sim_masked.item(),
            "mse": mse_masked.item(),
            "gen_norm": (gen_norm * gt_steps_mask).sum().item() / (gt_steps_mask.sum().item() + 1e-8),
            "gt_norm": (gt_norm * gt_steps_mask).sum().item() / (gt_steps_mask.sum().item() + 1e-8),
            "norm_ratio": norm_ratio_masked.item(),
        }

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
            return_latent_hidden_states: 是否返回中间状态
            
        Returns:
            pred_ids: 预测的token ids
            n_latent_forward: Latent步数
        """
        answer_generation_config = self.model_kwargs.answer_generation_config
        batch_size = len(questions)

        # 1. Question -> Embedding (不经过LLM forward)
        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        query_embedding = self.embedding(question_input_ids)  # [B, L_q, H]
        query_mask = question_attention_mask.float()  # [B, L_q]

        # 2. Diffusion生成Steps Embedding
        if self.use_cfg:
            steps_embeds = self.latent_diffusion.generate_with_cfg(
                condition=query_embedding,
                num_inference_steps=self.num_inference_steps,
                latent_length=self.max_latent_length,
                cfg_scale=self.cfg_scale,
                condition_mask=query_mask,
                clamp_value=self.clamp_value,
            )
        else:
            steps_embeds = self.latent_diffusion.generate(
                condition=query_embedding,
                num_inference_steps=self.num_inference_steps,
                latent_length=self.max_latent_length,
                condition_mask=query_mask,
                clamp_value=self.clamp_value,
            )

        # 3. 拼接并生成答案
        # 添加 ### 分隔符
        sep_ids = torch.full(
            (batch_size, 1), self.thinking_separator_id, device=self.device, dtype=torch.long
        )
        sep_embeds = self.embedding(sep_ids)

        all_embeds = torch.cat([query_embedding, steps_embeds, sep_embeds], dim=1)
        all_attention_mask = torch.cat(
            [
                question_attention_mask,
                torch.ones(batch_size, self.max_latent_length, device=self.device, dtype=question_attention_mask.dtype),
                torch.ones(batch_size, 1, device=self.device, dtype=question_attention_mask.dtype),
            ],
            dim=1,
        )

        # 4. 生成答案
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

    def on_fit_start(self):
        """训练开始时初始化CSV文件和归一化参数"""
        super().on_fit_start()
        
        # 初始化CSV
        if self.trainer.global_rank == 0:
            log_dir = self.trainer.logger.log_dir if self.trainer.logger else "logs/difflar"
            os.makedirs(log_dir, exist_ok=True)
            self.loss_csv_path = os.path.join(log_dir, "loss_history.csv")
            with open(self.loss_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'total_loss', 'diffusion_loss', 'answer_loss'])
        
        # 估计归一化参数（关键步骤！）
        if self.latent_diffusion.normalize_latent and not self._latent_stats_estimated:
            self._estimate_latent_stats(self.trainer.datamodule)

    def _estimate_latent_stats(self, datamodule=None):
        """从训练数据估计embedding的归一化参数"""
        print("Estimating latent statistics from training data...")
        
        # 获取dataloader
        if datamodule is not None:
            train_loader = datamodule.train_dataloader()
        elif hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            train_loader = self.trainer.datamodule.train_dataloader()
        else:
            print("Warning: Cannot get train_dataloader, skipping latent stats estimation")
            return
        
        # 收集一批数据来估计统计量
        all_embeds = []
        all_masks = []
        num_samples = 0
        max_samples = 500  # 用500个样本估计
        
        with torch.no_grad():
            for batch in train_loader:
                steps = batch["steps"]
                steps_embeds, steps_mask = self.prepare_fixed_length_steps(steps)
                all_embeds.append(steps_embeds.cpu())
                all_masks.append(steps_mask.cpu())
                num_samples += len(steps)
                if num_samples >= max_samples:
                    break
        
        # 合并并估计统计量
        all_embeds = torch.cat(all_embeds, dim=0).to(self.device)
        all_masks = torch.cat(all_masks, dim=0).to(self.device)
        
        mean, std, scale = self.latent_diffusion.estimate_latent_stats(all_embeds, all_masks)
        
        self._latent_stats_estimated = True
        print(f"Latent stats estimated:")
        print(f"  Mean norm: {mean.norm():.4f}")
        print(f"  Std mean: {std.mean():.6f}")
        print(f"  Scale: {scale:.4f}")

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        """重写training_step以记录Loss"""
        log_dict = self.forward(batch=batch)
        
        # 记录Loss到历史
        if self.trainer.global_rank == 0:
            step = self.global_step
            loss_record = {
                'step': step,
                'total_loss': log_dict['total_loss'].item(),
                'diffusion_loss': log_dict['diffusion_loss'].item(),
                'answer_loss': log_dict['answer_loss'].item(),
            }
            self.loss_history.append(loss_record)
            
            # 实时写入CSV
            if self.loss_csv_path:
                with open(self.loss_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, loss_record['total_loss'], 
                                   loss_record['diffusion_loss'], loss_record['answer_loss']])
        
        # 添加前缀并记录
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        self.log_dict(log_dict, sync_dist=True, prog_bar=True, 
                     batch_size=self.all_config.dataloader.batch_size)
        return log_dict["train/total_loss"]

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
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total Loss
        axes[0, 0].plot(steps, total_loss, 'b-', alpha=0.7)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Diffusion Loss
        axes[0, 1].plot(steps, diffusion_loss, 'g-', alpha=0.7)
        axes[0, 1].set_title('Diffusion Loss')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Answer Loss
        axes[1, 0].plot(steps, answer_loss, 'r-', alpha=0.7)
        axes[1, 0].set_title('Answer Loss')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # All Losses
        axes[1, 1].plot(steps, total_loss, 'b-', label='Total', alpha=0.7)
        axes[1, 1].plot(steps, diffusion_loss, 'g-', label='Diffusion', alpha=0.7)
        axes[1, 1].plot(steps, answer_loss, 'r-', label='Answer', alpha=0.7)
        axes[1, 1].set_title('All Losses')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        log_dir = self.trainer.logger.log_dir if self.trainer.logger else "logs/difflar"
        save_path = os.path.join(log_dir, "loss_curves.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss curves saved to: {save_path}")
