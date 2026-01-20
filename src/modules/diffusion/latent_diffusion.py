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
    1. 训练时：预测加噪Steps Embedding中的噪声/速度
    2. 推理时：从纯噪声生成Steps Embedding
    
    关键改进：
    - 数据归一化：将embedding归一化到标准正态分布，解决训练-推理分布偏移问题
    - 支持预测x0模式：更稳定的预测目标
    - Self-Conditioning：将上一步预测的x0作为额外条件，提高去噪质量
    - Flow Matching：线性插值路径 + 速度预测，避免数值爆炸
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
        prediction_type: str = "epsilon",  # "epsilon", "x0", or "flow"
        sampler_type: str = "ddpm",  # "ddpm", "ddim", or "euler" (for flow)
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
        
        # Flow Matching: 使用连续时间 [0, 1]
        self.use_flow_matching = (prediction_type == "flow")
        
        # Self-Conditioning配置
        self.self_cond_prob = 0.5  # 训练时Self-Cond概率

        # 归一化参数（将在首次forward时从数据估计，或手动设置）
        # 使用register_buffer确保保存到checkpoint
        self.register_buffer("latent_mean", torch.zeros(hidden_size))
        self.register_buffer("latent_std", torch.ones(hidden_size))
        self.register_buffer("latent_scale", torch.tensor(1.0))  # 全局缩放因子
        # 使用buffer保存初始化状态，确保checkpoint加载后状态正确
        self.register_buffer("_latent_stats_initialized", torch.tensor(False))

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
        self._latent_stats_initialized.fill_(True)

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
        use_self_cond: Optional[bool] = None,
        t_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练时的前向传播（支持Self-Conditioning和Flow Matching）
        
        Args:
            steps_embeds: 目标Steps Embedding [B, L, H]
            condition: Query Embedding [B, L_q, H]
            attention_mask: Steps的注意力掩码 [B, L]
            condition_mask: Query的注意力掩码 [B, L_q]
            use_self_cond: 是否使用Self-Conditioning，None表示按概率决定
            
        Returns:
            loss: 扩散损失
            predicted: 预测值（噪声、x0或速度，取决于prediction_type）
            target: 目标值
        """
        import random
        batch_size = steps_embeds.shape[0]
        device = steps_embeds.device

        # 0. 归一化（关键步骤！）
        x_0 = self.normalize(steps_embeds)

        # ═══ Flow Matching 路径 ═══
        if self.use_flow_matching:
            return self._forward_flow_matching(x_0, condition, attention_mask, condition_mask, use_self_cond, t_range=t_range)
        
        # ═══ DDPM/DDIM 路径 ═══
        # 1. 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # 2. 前向扩散：添加噪声（关键改进：只在有效位置加噪声）
        if attention_mask is not None:
            x_t, noise = self._masked_add_noise(x_0, t, attention_mask)
        else:
            x_t, noise = self.scheduler.add_noise(x_0, t)

        # 3. Self-Conditioning: 按概率决定是否使用
        if use_self_cond is None:
            use_self_cond = random.random() < self.self_cond_prob
        
        x0_cond = None
        if use_self_cond:
            with torch.no_grad():
                prev_output = self.denoiser(x_t, t, condition, attention_mask, condition_mask, x0_cond=None)
                if self.prediction_type == "epsilon":
                    x0_cond = self._get_x0_from_noise(x_t, prev_output, t)
                else:
                    x0_cond = prev_output

        # 4. 模型预测（带Self-Conditioning）
        model_output = self.denoiser(x_t, t, condition, attention_mask, condition_mask, x0_cond=x0_cond)

        # 5. 计算损失（根据预测类型）
        if self.prediction_type == "epsilon":
            target = noise
            loss = self.compute_loss(model_output, target, attention_mask)
        elif self.prediction_type == "x0":
            target = x_0
            loss = self.compute_loss(model_output, target, attention_mask)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return loss, model_output, target
    
    def _forward_flow_matching(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
        use_self_cond: Optional[bool],
        t_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flow Matching训练：线性插值 + 速度预测
        
        核心公式：
        - 插值: x_t = (1-t)*x_0 + t*noise
        - 目标速度: u = noise - x_0
        """
        import random
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 1. 采样连续时间 t
        # 默认：t ~ U[0, 1]
        # Stage1b高噪声训练可传入 t_range=(0.8, 1.0)
        if t_range is not None:
            t_min, t_max = t_range
            # 防御性处理，避免非法范围
            t_min = float(max(0.0, min(1.0, t_min)))
            t_max = float(max(0.0, min(1.0, t_max)))
            if t_max < t_min:
                t_min, t_max = t_max, t_min
            # 退化情况：范围极小
            if (t_max - t_min) < 1e-8:
                t = torch.full((batch_size,), t_min, device=device)
            else:
                t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
        else:
            t = torch.rand(batch_size, device=device)
        
        # 2. 采样噪声（关键改进：只在有效位置添加噪声）
        noise = torch.randn_like(x_0)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)  # [B, L, 1]
            noise = noise * mask_expanded  # padding位置噪声为0
        
        # 3. 线性插值构造 x_t（关键！避免复杂的alpha调度）
        # x_t = (1-t)*x_0 + t*noise
        t_expand = t.view(-1, 1, 1)  # [B, 1, 1]
        x_t = (1 - t_expand) * x_0 + t_expand * noise
        
        # 4. 目标速度（从数据指向噪声）
        # 对于padding位置：x_0=0, noise=0，所以velocity=0
        target_velocity = noise - x_0
        
        # 5. Self-Conditioning
        if use_self_cond is None:
            use_self_cond = random.random() < self.self_cond_prob
        
        x0_cond = None
        if use_self_cond:
            with torch.no_grad():
                # 将连续时间t转换为离散时间步（用于denoiser的时间编码）
                t_discrete = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)
                prev_velocity = self.denoiser(x_t, t_discrete, condition, attention_mask, condition_mask, x0_cond=None)
                # 精确公式: x_t = (1-t)*x_0 + t*noise, v = noise - x_0
                # 推导: x_0 = (x_t - t*v) / (1-t)
                x0_cond = (x_t - t_expand * prev_velocity) / (1 - t_expand + 1e-6)
        
        # 6. 模型预测速度
        t_discrete = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)
        model_output = self.denoiser(x_t, t_discrete, condition, attention_mask, condition_mask, x0_cond=x0_cond)
        
        # 7. 计算损失
        loss = self.compute_loss(model_output, target_velocity, attention_mask)
        
        return loss, model_output, target_velocity

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
        clamp_value: Optional[float] = 3.0,
        use_self_cond: bool = True,
    ) -> torch.Tensor:
        """
        推理时生成Steps Embedding（支持Self-Conditioning和Flow Matching）
        
        Args:
            condition: Query Embedding [B, L_q, H]
            num_inference_steps: 推理采样步数
            latent_length: 生成的Latent长度
            attention_mask: Steps的注意力掩码 [B, L]
            condition_mask: Query的注意力掩码 [B, L_q]
            clamp_value: 截断异常值的阈值（防止数值爆炸）
            use_self_cond: 是否使用Self-Conditioning（推荐开启）
            
        Returns:
            generated_embeds: 生成的Steps Embedding [B, L, H]（已反归一化）
        """
        batch_size = condition.shape[0]
        device = condition.device

        if latent_length is None:
            latent_length = self.max_latent_length

        context = torch.enable_grad() if enable_grad else torch.no_grad()
        
        with context:
            # 1. 从纯噪声开始（标准正态分布）
            x_t = torch.randn(batch_size, latent_length, self.hidden_size, device=device)
            
            # ═══ Flow Matching: 欧拉积分 ═══
            if self.use_flow_matching:
                x_0 = self._generate_flow_matching(
                    x_t, condition, num_inference_steps, 
                    attention_mask, condition_mask, use_self_cond, clamp_value
                )
                return self.denormalize(x_0)
            
            # ═══ DDPM/DDIM 采样 ═══
            x0_pred = None
            timesteps = self.scheduler.get_timesteps(num_inference_steps).to(device)

            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1)
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)

                x0_cond = x0_pred if use_self_cond else None
                model_output = self.denoiser(x_t, t_batch, condition, attention_mask, condition_mask, x0_cond=x0_cond)
                
                if use_self_cond:
                    if self.prediction_type == "epsilon":
                        x0_pred = self._get_x0_from_noise(x_t, model_output, t_batch)
                    else:
                        x0_pred = model_output
                    if clamp_value is not None:
                        x0_pred = torch.clamp(x0_pred, -clamp_value, clamp_value)

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

            x_0 = self.denormalize(x_t)
            return x_0
    
    def _generate_flow_matching(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        num_inference_steps: int,
        attention_mask: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
        use_self_cond: bool,
        clamp_value: Optional[float],
    ) -> torch.Tensor:
        """
        Flow Matching推理：欧拉积分从噪声走到数据
        
        核心公式：x_{t-dt} = x_t - dt * velocity
        从t=1（噪声）积分到t=0（数据）
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # 时间步：从1到0均匀分布
        dt = 1.0 / num_inference_steps
        timesteps = torch.linspace(1.0, dt, num_inference_steps, device=device)
        
        x0_pred = None
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=device)
            t_discrete = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)
            
            # Self-Conditioning
            x0_cond = x0_pred if use_self_cond else None
            
            # 预测速度
            velocity = self.denoiser(x_t, t_discrete, condition, attention_mask, condition_mask, x0_cond=x0_cond)
            
            # 精确公式: x_t = (1-t)*x_0 + t*noise, v = noise - x_0
            # 推导: x_0 = (x_t - t*v) / (1-t)
            if use_self_cond:
                t_expand = t.view(-1, 1, 1)
                x0_pred = (x_t - t_expand * velocity) / (1 - t_expand + 1e-6)
                if clamp_value is not None:
                    x0_pred = torch.clamp(x0_pred, -clamp_value, clamp_value)
            
            # 欧拉积分：向数据方向移动
            # x_{t-dt} = x_t - dt * velocity（从噪声方向减去，向数据方向走）
            x_t = x_t - dt * velocity
            
            if clamp_value is not None:
                x_t = torch.clamp(x_t, -clamp_value * 2, clamp_value * 2)
        
        return x_t
    
    def _get_x0_from_noise(self, x_t: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """从预测的噪声反推x_0"""
        sqrt_alpha_cumprod = self.scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        # x_0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
        x_0 = (x_t - sqrt_one_minus_alpha_cumprod * noise) / (sqrt_alpha_cumprod + 1e-8)
        return x_0

    def _get_noise_from_x0(self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """从预测的x_0反推噪声"""
        sqrt_alpha_cumprod = self.scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        noise = (x_t - sqrt_alpha_cumprod * x_0) / (sqrt_one_minus_alpha_cumprod + 1e-8)
        return noise
    
    def _get_noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """获取时间步t对应的噪声水平"""
        return self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
    
    def _masked_add_noise(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        带mask的加噪声：只在有效位置添加噪声，padding位置保持为零
        
        这解决了变长序列的核心问题：
        - 有效位置：正常加噪声进行扩散训练
        - Padding位置：保持为零，不参与扩散过程
        """
        noise = torch.randn_like(x_0)
        mask_expanded = attention_mask.unsqueeze(-1)  # [B, L, 1]
        
        # 只在有效位置添加噪声
        noise = noise * mask_expanded
        
        # 使用scheduler的系数计算x_t
        sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * noise
        # 对于padding位置：x_0=0, noise=0，所以x_t=0
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise

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
        use_self_cond: bool = True,
    ) -> torch.Tensor:
        """
        使用Classifier-Free Guidance的生成（支持Self-Conditioning和Flow Matching）
        """
        batch_size = condition.shape[0]
        device = condition.device

        if latent_length is None:
            latent_length = self.max_latent_length

        context = torch.enable_grad() if enable_grad else torch.no_grad()
        
        with context:
            x_t = torch.randn(batch_size, latent_length, self.hidden_size, device=device)
            uncond = torch.zeros_like(condition)
            
            # ═══ Flow Matching + CFG ═══
            if self.use_flow_matching:
                x_0 = self._generate_flow_matching_cfg(
                    x_t, condition, uncond, num_inference_steps, cfg_scale,
                    attention_mask, condition_mask, use_self_cond, clamp_value
                )
                return self.denormalize(x_0)
            
            # ═══ DDPM/DDIM + CFG ═══
            x0_pred = None
            timesteps = self.scheduler.get_timesteps(num_inference_steps).to(device)

            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1)
                t_prev_batch = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)

                x0_cond = x0_pred if use_self_cond else None
                output_cond = self.denoiser(x_t, t_batch, condition, attention_mask, condition_mask, x0_cond=x0_cond)
                output_uncond = self.denoiser(x_t, t_batch, uncond, attention_mask, condition_mask, x0_cond=x0_cond)

                model_output = output_uncond + cfg_scale * (output_cond - output_uncond)
                
                if use_self_cond:
                    if self.prediction_type == "epsilon":
                        x0_pred = self._get_x0_from_noise(x_t, model_output, t_batch)
                    else:
                        x0_pred = model_output
                    if clamp_value is not None:
                        x0_pred = torch.clamp(x0_pred, -clamp_value, clamp_value)

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
                        add_noise = (i < len(timesteps) - 1)
                        x_t = self.scheduler.denoise_step(x_t, model_output, t_batch, add_noise)
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
    
    def _generate_flow_matching_cfg(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor,
        uncond: torch.Tensor,
        num_inference_steps: int,
        cfg_scale: float,
        attention_mask: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
        use_self_cond: bool,
        clamp_value: Optional[float],
    ) -> torch.Tensor:
        """Flow Matching + CFG推理"""
        batch_size = x_t.shape[0]
        device = x_t.device
        
        dt = 1.0 / num_inference_steps
        timesteps = torch.linspace(1.0, dt, num_inference_steps, device=device)
        
        x0_pred = None
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=device)
            t_discrete = (t * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)
            
            x0_cond = x0_pred if use_self_cond else None
            
            # CFG: 有条件和无条件预测
            velocity_cond = self.denoiser(x_t, t_discrete, condition, attention_mask, condition_mask, x0_cond=x0_cond)
            velocity_uncond = self.denoiser(x_t, t_discrete, uncond, attention_mask, condition_mask, x0_cond=x0_cond)
            
            velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
            
            if use_self_cond:
                t_expand = t.view(-1, 1, 1)
                x0_pred = (x_t - t_expand * velocity) / (1 - t_expand + 1e-6)
                if clamp_value is not None:
                    x0_pred = torch.clamp(x0_pred, -clamp_value, clamp_value)
            
            x_t = x_t - dt * velocity
            
            if clamp_value is not None:
                x_t = torch.clamp(x_t, -clamp_value * 2, clamp_value * 2)
        
        return x_t
