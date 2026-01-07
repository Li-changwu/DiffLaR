import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .scheduler import NoiseScheduler
from .denoiser import Denoiser


class LatentDiffusion(nn.Module):
    """
    Latent Diffusion模型：在LLM的Embedding空间进行扩散和去噪
    
    核心功能：
    1. 训练时：预测加噪Steps Embedding中的噪声
    2. 推理时：从纯噪声生成Steps Embedding
    
    关键改进：
    - 数据归一化：将embedding归一化到标准正态分布，解决训练-推理分布偏移问题
    - 支持预测x0模式：更稳定的预测目标
    """

    def __init__(
        self,
        hidden_size: int,
        num_timesteps: int = 1000,
        num_layers: int = 6,
        num_heads: int = 8,
        max_latent_length: int = 256,
        schedule_type: str = "linear",
        dropout: float = 0.1,
        normalize_latent: bool = True,
        prediction_type: str = "epsilon",  # "epsilon" or "x0"
        sampler_type: str = "ddpm",  # "ddpm" or "ddim"
        ddim_eta: float = 0.0,  # DDIM随机性，0=确定性
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_latent_length = max_latent_length
        self.num_timesteps = num_timesteps
        self.normalize_latent = normalize_latent
        self.prediction_type = prediction_type
        self.sampler_type = sampler_type
        self.ddim_eta = ddim_eta

        # 归一化参数（将在首次forward时从数据估计，或手动设置）
        # 使用register_buffer确保保存到checkpoint
        self.register_buffer("latent_mean", torch.zeros(hidden_size))
        self.register_buffer("latent_std", torch.ones(hidden_size))
        self.register_buffer("latent_scale", torch.tensor(1.0))  # 全局缩放因子
        self.latent_stats_initialized = False

        # 噪声调度器
        self.scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            schedule_type=schedule_type,
        )

        # 去噪网络
        self.denoiser = Denoiser(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_length=max_latent_length,
        )

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor, scale: float = 1.0):
        """手动设置归一化参数"""
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)
        self.latent_scale.fill_(scale)
        self.latent_stats_initialized = True

    def estimate_latent_stats(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        从数据估计归一化参数
        Args:
            embeddings: [B, L, H] 或 [N, H]
            mask: [B, L] 有效位置掩码
        """
        if embeddings.dim() == 3:
            if mask is not None:
                # 只统计有效位置
                mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
                valid_embeds = embeddings[mask_expanded.bool()].view(-1, self.hidden_size)
            else:
                valid_embeds = embeddings.view(-1, self.hidden_size)
        else:
            valid_embeds = embeddings
        
        mean = valid_embeds.mean(dim=0)
        std = valid_embeds.std(dim=0).clamp(min=1e-6)
        # 计算全局缩放因子，使归一化后的数据范数接近sqrt(hidden_size)
        normalized = (valid_embeds - mean) / std
        current_norm = normalized.norm(dim=-1).mean()
        target_norm = (self.hidden_size ** 0.5)  # 标准正态的期望范数
        scale = target_norm / (current_norm + 1e-6)
        
        self.set_latent_stats(mean, std, scale.item())
        return mean, std, scale

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """归一化embedding到标准分布"""
        if not self.normalize_latent:
            return x
        x_normalized = (x - self.latent_mean) / self.latent_std
        return x_normalized * self.latent_scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化，恢复到原始尺度"""
        if not self.normalize_latent:
            return x
        x_unscaled = x / self.latent_scale
        return x_unscaled * self.latent_std + self.latent_mean

    def forward(
        self,
        steps_embeds: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练时的前向传播
        
        Args:
            steps_embeds: 目标Steps Embedding [B, L, H]
            condition: Query Embedding [B, L_q, H]
            attention_mask: Steps的注意力掩码 [B, L]
            condition_mask: Query的注意力掩码 [B, L_q]
            
        Returns:
            loss: 扩散损失
            predicted: 预测值（噪声或x0，取决于prediction_type）
            target: 目标值
        """
        batch_size = steps_embeds.shape[0]
        device = steps_embeds.device

        # 0. 归一化（关键步骤！）
        x_0 = self.normalize(steps_embeds)

        # 1. 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # 2. 前向扩散：添加噪声
        x_t, noise = self.scheduler.add_noise(x_0, t)

        # 3. 模型预测
        model_output = self.denoiser(x_t, t, condition, attention_mask, condition_mask)

        # 4. 计算损失（根据预测类型）
        if self.prediction_type == "epsilon":
            target = noise
            loss = self.compute_loss(model_output, target, attention_mask)
        elif self.prediction_type == "x0":
            target = x_0
            loss = self.compute_loss(model_output, target, attention_mask)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return loss, model_output, target

    def compute_loss(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算扩散损失（MSE）"""
        loss = F.mse_loss(predicted_noise, target_noise, reduction="none")
        loss = loss.mean(dim=-1)  # [B, L]

        if attention_mask is not None:
            # 只计算有效位置的损失
            loss = (loss * attention_mask).sum() / (attention_mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def generate(
        self,
        condition: torch.Tensor,
        num_inference_steps: int = 128,
        latent_length: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        enable_grad: bool = False,
        clamp_value: Optional[float] = 3.0,  # 截断异常值
    ) -> torch.Tensor:
        """
        推理时生成Steps Embedding
        
        Args:
            condition: Query Embedding [B, L_q, H]
            num_inference_steps: 推理采样步数
            latent_length: 生成的Latent长度
            attention_mask: Steps的注意力掩码 [B, L]
            condition_mask: Query的注意力掩码 [B, L_q]
            clamp_value: 截断异常值的阈值（防止数值爆炸）
            
        Returns:
            generated_embeds: 生成的Steps Embedding [B, L, H]（已反归一化）
        """
        batch_size = condition.shape[0]
        device = condition.device

        if latent_length is None:
            latent_length = self.max_latent_length

        # 根据是否需要梯度选择上下文
        context = torch.enable_grad() if enable_grad else torch.no_grad()
        
        with context:
            # 1. 从纯噪声开始（标准正态分布）
            x_t = torch.randn(batch_size, latent_length, self.hidden_size, device=device)

            # 2. 获取推理时间步
            timesteps = self.scheduler.get_timesteps(num_inference_steps).to(device)

            # 3. 逐步去噪
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1)
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)

                # 模型预测
                model_output = self.denoiser(x_t, t_batch, condition, attention_mask, condition_mask)

                # 根据采样器类型和预测类型进行去噪
                if self.sampler_type == "ddim":
                    # DDIM采样（更稳定）
                    if self.prediction_type == "epsilon":
                        x_t = self.scheduler.ddim_step(x_t, model_output, t_batch, t_prev_batch, self.ddim_eta)
                    else:  # x0
                        pred_x0 = model_output
                        if clamp_value is not None:
                            pred_x0 = torch.clamp(pred_x0, -clamp_value, clamp_value)
                        x_t = self.scheduler.ddim_step_from_x0(x_t, pred_x0, t_batch, t_prev_batch, self.ddim_eta)
                else:
                    # DDPM采样
                    if self.prediction_type == "epsilon":
                        predicted_noise = model_output
                        add_noise = (i < len(timesteps) - 1)
                        x_t = self.scheduler.denoise_step(x_t, predicted_noise, t_batch, add_noise)
                    elif self.prediction_type == "x0":
                        pred_x0 = model_output
                        if clamp_value is not None:
                            pred_x0 = torch.clamp(pred_x0, -clamp_value, clamp_value)
                        predicted_noise = self._get_noise_from_x0(x_t, pred_x0, t_batch)
                        add_noise = (i < len(timesteps) - 1)
                        x_t = self.scheduler.denoise_step(x_t, predicted_noise, t_batch, add_noise)
                
                # 截断x_t防止累积误差爆炸
                if clamp_value is not None:
                    x_t = torch.clamp(x_t, -clamp_value * 2, clamp_value * 2)

            # 4. 反归一化到原始尺度
            x_0 = self.denormalize(x_t)
            return x_0

    def _get_noise_from_x0(self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """从预测的x_0反推噪声"""
        sqrt_alpha_cumprod = self.scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        # noise = (x_t - sqrt(alpha_bar) * x_0) / sqrt(1-alpha_bar)
        noise = (x_t - sqrt_alpha_cumprod * x_0) / (sqrt_one_minus_alpha_cumprod + 1e-8)
        return noise

    def generate_with_cfg(
        self,
        condition: torch.Tensor,
        num_inference_steps: int = 128,
        latent_length: Optional[int] = None,
        cfg_scale: float = 1.5,
        attention_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        enable_grad: bool = False,
        clamp_value: Optional[float] = 3.0,
    ) -> torch.Tensor:
        """
        使用Classifier-Free Guidance的生成
        
        Args:
            condition: Query的条件向量 [B, 1, H]
            num_inference_steps: 推理采样步数
            latent_length: 生成的Latent长度
            cfg_scale: CFG权重
            attention_mask: 注意力掩码 [B, L]
            clamp_value: 截断阈值
            
        Returns:
            generated_embeds: 生成的Steps Embedding [B, L, H]（已反归一化）
        """
        batch_size = condition.shape[0]
        device = condition.device

        if latent_length is None:
            latent_length = self.max_latent_length

        context = torch.enable_grad() if enable_grad else torch.no_grad()
        
        with context:
            # 1. 从纯噪声开始
            x_t = torch.randn(batch_size, latent_length, self.hidden_size, device=device)

            # 2. 无条件向量
            uncond = torch.zeros_like(condition)

            # 3. 获取推理时间步
            timesteps = self.scheduler.get_timesteps(num_inference_steps).to(device)

            # 4. 逐步去噪
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1)
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)

                # 有条件和无条件预测
                output_cond = self.denoiser(x_t, t_batch, condition, attention_mask, condition_mask)
                output_uncond = self.denoiser(x_t, t_batch, uncond, attention_mask, condition_mask)

                # CFG组合
                model_output = output_uncond + cfg_scale * (output_cond - output_uncond)

                # 根据采样器类型和预测类型去噪
                if self.sampler_type == "ddim":
                    if self.prediction_type == "epsilon":
                        x_t = self.scheduler.ddim_step(x_t, model_output, t_batch, t_prev_batch, self.ddim_eta)
                    else:
                        pred_x0 = model_output
                        if clamp_value is not None:
                            pred_x0 = torch.clamp(pred_x0, -clamp_value, clamp_value)
                        x_t = self.scheduler.ddim_step_from_x0(x_t, pred_x0, t_batch, t_prev_batch, self.ddim_eta)
                else:
                    if self.prediction_type == "epsilon":
                        predicted_noise = model_output
                        add_noise = (i < len(timesteps) - 1)
                        x_t = self.scheduler.denoise_step(x_t, predicted_noise, t_batch, add_noise)
                    elif self.prediction_type == "x0":
                        pred_x0 = model_output
                        if clamp_value is not None:
                            pred_x0 = torch.clamp(pred_x0, -clamp_value, clamp_value)
                        predicted_noise = self._get_noise_from_x0(x_t, pred_x0, t_batch)
                        add_noise = (i < len(timesteps) - 1)
                        x_t = self.scheduler.denoise_step(x_t, predicted_noise, t_batch, add_noise)
                
                if clamp_value is not None:
                    x_t = torch.clamp(x_t, -clamp_value * 2, clamp_value * 2)

            # 反归一化
            x_0 = self.denormalize(x_t)
            return x_0
