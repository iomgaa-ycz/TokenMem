"""LinearFusion —— 低秩门控融合模块（照搬 DecoupledRAG）。

核心公式: output = A + alpha * dropout(B) @ W_A @ W_B
- W_A: Gaussian σ=0.01 初始化
- W_B: Zero 初始化 → t=0 时 fusion 输出为零，不干扰 base LLM
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


class LinearFusion(nn.Module):
    """低秩门控融合: output = A + alpha * dropout(B) @ W_A @ W_B。

    参数：
        hidden_dim: 隐层维度
        rank: 低秩分解秩 (默认 16)
        alpha: 缩放因子 (默认 32)
        dropout_prob: knowledge dropout (默认 0.2)
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 16,
        alpha: int = 32,
        dropout_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.W_A = Parameter(torch.randn(hidden_dim, rank) * 0.01)
        self.W_B = Parameter(torch.zeros(rank, hidden_dim))
        self.rank = rank
        self.alpha = alpha
        self.dropout_prob = dropout_prob

    def forward(self, A: Tensor, B: Tensor) -> Tensor:
        """门控融合。

        参数：
            A: residual (self-attn + FFN 输出) [B, L, D]
            B: cross-attention 输出 [B, L, D]
        返回：
            融合后的 hidden states [B, L, D]
        """
        dtype = A.dtype
        A = A.to(self.W_A.dtype)
        B = B.to(self.W_A.dtype)
        B = F.dropout(B, p=self.dropout_prob, training=self.training)
        C = A + self.alpha * torch.matmul(torch.matmul(B, self.W_A), self.W_B)
        return C.to(dtype)
