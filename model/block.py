import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.norms import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model,n_head, d_ff = None, dropout = 0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        # Multi Head Attention
        x = x + self.attn(self.norm1(x)) # residual connection et RMS Norm (pre-norm)

        # Feed forward
        x = x + self.ffn(self.norm2(x)) # residual connection et RMS Norm (pre-norm)

        return x

