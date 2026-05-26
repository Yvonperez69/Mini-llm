import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff = None, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_ff = int(round((8/3 * d_model) / 64) * 64)
        self.w1 = nn.Linear(d_model, self.d_ff)
        self.w2 = nn.Linear(d_model, self.d_ff)
        self.w3 = nn.Linear(self.d_ff, d_model)
        self.SiLU = nn.SiLU()
        
    def forward(self, x):
        x = self.w3(self.SiLU(self.w1(x))*self.w2(x))
        return x
