import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff = None, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.ffn(x)
