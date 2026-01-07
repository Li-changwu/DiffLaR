import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    """时间步的正弦位置编码"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class Denoiser(nn.Module):
    """
    去噪网络：基于Transformer的噪声预测模型
    输入：加噪的Steps Embedding + 时间步 + Query条件
    输出：预测的噪声
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 256,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # 时间步编码
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # Cross-Attention: 用于融合Query Embedding条件
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)

        # 序列位置编码
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)

        # 输入投影
        self.input_proj = nn.Linear(hidden_size, hidden_size)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        # 自适应LayerNorm参数（用于融合时间步信息）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: 加噪的Steps Embedding [B, L, H]
            t: 时间步 [B]
            condition: Query Embedding [B, L_q, H] 变长序列
            attention_mask: Steps的注意力掩码 [B, L]
            condition_mask: Query的注意力掩码 [B, L_q]
            
        Returns:
            predicted_noise: 预测的噪声 [B, L, H]
        """
        batch_size, seq_len, _ = x_t.shape

        # 1. 时间步编码
        t_emb = self.time_embedding(t.float())  # [B, H]

        # 2. 计算自适应LayerNorm参数 (只用时间步)
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)  # [B, H] each

        # 3. 输入处理
        x = self.input_proj(x_t)  # [B, L, H]

        # 4. 添加位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)

        # 5. 应用自适应调制 (时间步信息)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 6. Cross-Attention: 融合Query条件
        # x作为query, condition作为key/value
        if condition_mask is not None:
            key_padding_mask = (condition_mask == 0)
        else:
            key_padding_mask = None
        
        x_normed = self.cross_attn_norm(x)
        cross_out, _ = self.cross_attn(
            query=x_normed,
            key=condition,
            value=condition,
            key_padding_mask=key_padding_mask,
        )
        x = x + cross_out  # 残差连接

        # 7. 创建self-attention mask
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # 8. Transformer编码 (Self-Attention)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 9. 输出投影
        predicted_noise = self.output_proj(x)

        return predicted_noise
