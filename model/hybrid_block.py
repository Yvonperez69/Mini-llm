from model.block import TransformerBlock
from model.sca import SCA
import torch.nn as nn 

class HybridBlock(nn.Module):
    def __init__(self, d_model, n_head_memory, n_head_query, spectral_sample, head_dim, n_head, n_kv_head, d_ff, context_length, dropout=0.1):
        super().__init__()
        self.sca = SCA(d_model, n_head_memory, n_head_query, spectral_sample, head_dim)
        self.transformer = TransformerBlock(d_model, n_head, n_kv_head, context_length, d_ff, dropout)
        
    def forward(self, x):
        x = self.sca(x)
        x = self.transformer(x)
        return x