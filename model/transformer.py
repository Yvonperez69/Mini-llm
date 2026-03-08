import torch
import torch.nn as nn 
import math
from model.block import TransformerBlock

class PositionalEmbidding(nn.Module):
    def __init__(self, d_model, max_len = 2048):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=float)*(-math.log(10000.0))/d_model)
        
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe =pe.unsqueeze(0) # Broadcast pour ajouter au embeddings des tokens (B,T,d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:,:T,:]

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.token_emb(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, d_ff = None, dropout = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Embeddings and positional encoding
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbidding(d_model)

        # Transformer blocks 
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])

        # final linear layer for output
        self.ln_f = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        x = self.output_layer(x)
        return x