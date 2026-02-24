import torch
import torch.nn as nn 
import math

class PositionalEmbidding(nn.Module):
    def __init__(self, d_model, max_len = 2048):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
                