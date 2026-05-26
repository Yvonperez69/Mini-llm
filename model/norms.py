import torch 
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        RMS = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x/RMS)*self.gamma