import torch.nn as nn 
from model.block import TransformerBlock
from model.norms import RMSNorm

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

        # Embeddings 
        self.token_emb = TokenEmbedding(vocab_size, d_model)

        # Transformer blocks 
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])

        # final linear layer for output
        self.ln_f = RMSNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        

    def forward(self, x):
        x = self.token_emb(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        x = self.output_layer(x)
        return x
