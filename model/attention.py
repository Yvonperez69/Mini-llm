import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # les projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x) :
            B, T, C = x.shape # batch size, sequence length, embedding dimension

            # projeter les entrées
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
            v = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
            k = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)

            # Scaled Dot-product Attention

            attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # masque
            mask = torch.tril(torch.ones(T,T,device=x.device)) # matrice de taille (T,T)
            mask = mask.unsqueeze(0).unsqueeze(0) # tenseur de taille (1,1,T,T)

            attention.masked_fill(mask==0, float("-inf")) # application du masque

            attn_proba = F.softmax(attention,dim=-1)
            
            out = attn_proba @ v
            out = out.transpose(1,2).contiguous().view(B,T,C)
            out = self.out_proj(out)

            return out