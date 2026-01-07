import torch
import torch.nn as nn
import math


class NoiseScheduler(nn.Module):
    """
    噪声调度器：管理前向扩散和反向去噪过程中的噪声级别
    采用线性噪声调度（参考LLaDA）
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # 注册为buffer，不参与梯度计算但会随模型保存
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def _cosine_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine噪声调度"""
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> tuple:
        """
        前向扩散：给原始数据添加噪声
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_0: 原始数据 [B, L, H]
            t: 时间步 [B]
            noise: 可选的预定义噪声
            
        Returns:
            x_t: 加噪后的数据
            noise: 使用的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise

    def denoise_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        反向去噪一步
        
        Args:
            x_t: 当前噪声数据 [B, L, H]
            predicted_noise: 模型预测的噪声 [B, L, H]
            t: 当前时间步 [B]
            add_noise: 是否添加随机噪声（最后一步不添加）
            
        Returns:
            x_{t-1}: 去噪后的数据
        """
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)

        # 计算预测的x_0方向
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
        )

        if add_noise and t.min() > 0:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x_t)
            x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x_prev = model_mean

        return x_prev

    def get_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """获取推理时的时间步序列"""
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = torch.flip(timesteps, [0])  # 从大到小
        return timesteps

    def ddim_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM采样步骤（确定性采样，更稳定）
        
        Args:
            x_t: 当前噪声数据 [B, L, H]
            predicted_noise: 预测的噪声 [B, L, H]
            t: 当前时间步 [B]
            t_prev: 下一个时间步 [B]
            eta: 随机性控制，0=完全确定性，1=DDPM
        """
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_cumprod_t_prev = torch.where(
            t_prev >= 0,
            self.alphas_cumprod[t_prev.clamp(min=0)],
            torch.ones_like(self.alphas_cumprod[0])
        ).view(-1, 1, 1)
        
        # 预测x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # 计算方差
        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
        )
        
        # 计算x_{t-1}的方向
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) * predicted_noise
        
        # DDIM更新
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev

    def ddim_step_from_x0(
        self,
        x_t: torch.Tensor,
        pred_x0: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM采样（从预测的x0出发）
        适用于prediction_type="x0"的情况
        """
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_cumprod_t_prev = torch.where(
            t_prev >= 0,
            self.alphas_cumprod[t_prev.clamp(min=0)],
            torch.ones_like(self.alphas_cumprod[0])
        ).view(-1, 1, 1)
        
        # 从pred_x0反推噪声
        predicted_noise = (x_t - torch.sqrt(alpha_cumprod_t) * pred_x0) / (torch.sqrt(1 - alpha_cumprod_t) + 1e-8)
        
        # 计算方差
        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t + 1e-8) * (1 - alpha_cumprod_t / (alpha_cumprod_t_prev + 1e-8))
        )
        
        # 计算方向
        dir_xt = torch.sqrt((1 - alpha_cumprod_t_prev - sigma_t ** 2).clamp(min=0)) * predicted_noise
        
        # DDIM更新
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev
