import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.norms import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head, context_length, d_ff = None, dropout = 0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_head,n_kv_head,context_length, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, kv_cache = None):
        # Multi Head Attention
        attn_out, new_kv_cache = self.attn(self.norm1(x), kv_cache)
        x = x + attn_out # residual connection et RMS Norm (pre-norm)

        # Feed forward
        x = x + self.ffn(self.norm2(x)) # residual connection et RMS Norm (pre-norm)

        # même signature que SCA : (x, state) -> (out, new_state)
        return x, new_kv_cache

