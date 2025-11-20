"""
Cross-Attention with LARoPE for Vector Estimator

Latent (Query) と Text (Key/Value) の間のCross-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.common import LARoPE


class CrossAttentionLARoPE(nn.Module):
    """
    Cross-Attention with LARoPE

    Query: Latent (音声)
    Key/Value: Text (テキスト)

    LARoPEを使用してテキスト長に応じた位置エンコーディングを適用
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        n_heads: int = 4,
        use_residual: bool = True,
        rotary_base: int = 10000,
        rotary_scale: float = 10.0,
    ):
        """
        引数:
            q_dim: Queryの次元
            kv_dim: Key/Valueの次元
            n_heads: ヘッド数
            use_residual: Residual接続を使用するか
            rotary_base: RoPEのベース値
            rotary_scale: RoPEのスケーリング係数
        """
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.n_heads = n_heads
        self.head_dim = q_dim // n_heads
        self.use_residual = use_residual

        assert q_dim % n_heads == 0, "q_dim must be divisible by n_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)  # Key: kv_dim → q_dim
        self.v_proj = nn.Linear(kv_dim, q_dim)  # Value: kv_dim → q_dim
        self.out_proj = nn.Linear(q_dim, q_dim)

        # Layer Normalization
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)

        # LARoPE
        self.rope = LARoPE(self.head_dim, rotary_base, rotary_scale)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
        text_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        順伝播

        引数:
            query: Query (latent) [batch, q_dim, q_seq_len]
            key_value: Key/Value (text) [batch, kv_dim, kv_seq_len]
            query_mask: Query mask [batch, 1, q_seq_len]
            kv_mask: Key/Value mask [batch, 1, kv_seq_len]
            text_len: テキスト長 (LARoPE用)

        戻り値:
            出力 [batch, q_dim, q_seq_len]
        """
        batch_size = query.size(0)
        q_seq_len = query.size(2)
        kv_seq_len = key_value.size(2)

        # [batch, dim, seq_len] -> [batch, seq_len, dim]
        query = query.transpose(1, 2)
        key_value = key_value.transpose(1, 2)

        # Residual
        residual = query if self.use_residual else 0

        # Layer Normalization
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)

        # Linear projections
        q = self.q_proj(query)  # [batch, q_seq_len, q_dim]
        k = self.k_proj(key_value)  # [batch, kv_seq_len, q_dim]
        v = self.v_proj(key_value)  # [batch, kv_seq_len, q_dim]

        # Reshape for multi-head attention
        # [batch, seq_len, q_dim] -> [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply LARoPE
        if text_len is not None:
            # Query用のRoPE (latent長に対して)
            cos_q, sin_q = self.rope(q_seq_len, text_len, query.device)
            q = LARoPE.apply_rotary_pos_emb(q, cos_q, sin_q)

            # Key用のRoPE (text長に対して)
            cos_k, sin_k = self.rope(kv_seq_len, text_len, key_value.device)
            k = LARoPE.apply_rotary_pos_emb(k, cos_k, sin_k)

        # Scaled dot-product attention
        # Q @ K^T
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [batch, n_heads, q_seq_len, kv_seq_len]
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        # Apply attention mask
        if kv_mask is not None:
            # kv_mask: [batch, 1, kv_seq_len] -> [batch, 1, 1, kv_seq_len]
            kv_mask = kv_mask.unsqueeze(2)
            attn_weights = attn_weights.masked_fill(kv_mask == 0, float("-inf"))

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attention @ V
        attn_output = torch.matmul(attn_weights, v)  # [batch, n_heads, q_seq_len, head_dim]

        # Reshape back
        # [batch, n_heads, q_seq_len, head_dim] -> [batch, q_seq_len, q_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_seq_len, self.q_dim)

        # Output projection
        output = self.out_proj(attn_output)

        # Residual connection
        output = output + residual

        # Apply query mask
        if query_mask is not None:
            query_mask = query_mask.transpose(1, 2)  # [batch, q_seq_len, 1]
            output = output * query_mask

        # [batch, q_seq_len, q_dim] -> [batch, q_dim, q_seq_len]
        output = output.transpose(1, 2)

        return output
