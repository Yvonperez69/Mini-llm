import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model,n_head, d_ff = None, dropout = 0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Multi Head Attention
        attn_out = self.attn(x)
        x = x + attn_out # residual connection
        x = self.norm1(x) # layer normalization

        # Feed forward
        ffn_out = self.ffn(x)
        x = x + ffn_out # residual connection
        x = self.norm2(x) # layer normalization

        return x