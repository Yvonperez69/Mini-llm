import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head, context_length, dropout=0.1):
        super().__init__()
        
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = d_model // n_head
        self.n_rep = n_head // n_kv_head

        # les projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_head*self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_head*self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        cos, sin = self._build_rope_cache(context_length,device=torch.device("cpu"))
        self.register_buffer("cos",cos)
        self.register_buffer("sin",sin)
        
    def _build_rope_cache(self, context_length, device):
        # fréquences : shape (head_dim//2,)
        theta = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=device) / self.head_dim))
        # positions : shape (T,)
        positions = torch.arange(context_length, device=device)
        # produit externe : shape (T, head_dim//2)
        freqs = torch.outer(positions, theta)
        # cos et sin : shape (1, 1, T, head_dim//2) pour broadcaster avec (B, n_head, T, head_dim)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin
        
    def _apply_rope(self, x, cos, sin):
        # séparer les paires de dimensions
        x1 = x[..., 0::2]  # dimensions paires
        x2 = x[..., 1::2]  # dimensions impaires
        # rotation
        x_rot = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)
        return x_rot

    def forward(self, x, kv_cache = None) :
        """kv_cache = (k, v) des tokens déjà traités, ou None en début de séquence.

        Sans cache, un appel avec T=1 n'attend que sur ce seul token et le place
        à la position 0 : la génération perd tout son historique.
        """
        B, T, C = x.shape # batch size, sequence length, embedding dimension

        # nombre de tokens déjà vus : décale le RoPE et élargit le masque
        past = 0 if kv_cache is None else kv_cache[0].shape[2]
        assert past + T <= self.cos.shape[2], (
            f"contexte dépassé : {past + T} positions demandées, "
            f"{self.cos.shape[2]} en cache RoPE"
        )

        # projeter les entrées
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1,2)

        # On applique la rotation du RoPE, aux positions absolues past..past+T
        cos = self.cos[:, :, past:past+T, :]
        sin = self.sin[:, :, past:past+T, :]
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Concaténation après le RoPE — il est absolu, donc les clés en cache
        # gardent la rotation de leur propre position — et avant le
        # repeat_interleave, pour ne stocker que n_kv_head têtes.
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k.detach(), v.detach())

        # répéter K et V pour chaque groupe de Q
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled Dot-product Attention

        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # masque rectangulaire : T requêtes contre past+T clés.
        # diagonal=past autorise la requête de la ligne i jusqu'à la colonne
        # i+past, qui est sa position absolue dans la séquence.
        mask = torch.ones(T, past + T, device=x.device).tril(diagonal=past)
        mask = mask.unsqueeze(0).unsqueeze(0) # tenseur de taille (1,1,T,past+T)

        attention = attention.masked_fill(mask==0, float("-inf")) # application du masque

        attn_proba = F.softmax(attention,dim=-1)
        attn_proba = self.dropout(attn_proba)

        out = attn_proba @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.out_proj(out)

        return out, new_kv_cache




