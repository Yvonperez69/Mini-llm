import math
import torch
import torch.nn as nn

from model.block import TransformerBlock


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (fixed, non-learned).
    Produces a (1, T, d_model) tensor added to token embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)  # not a parameter, saved with state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        return: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class GPT(nn.Module):
    """
    Minimal decoder-only Transformer (GPT-like) for next-token prediction.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_head: int = 8,
        d_ff: int | None = None,
        max_len: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            # weight tying: output projection shares weights with input embeddings
            self.lm_head.weight = self.tok_emb.weight

        # init (simple, workable)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) token ids
        returns logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} > max_len {self.max_len}")

        x = self.tok_emb(idx)               # (B, T, d_model)
        x = self.pos_enc(x)                 # (B, T, d_model)
        x = self.drop(x)                    # (B, T, d_model)

        for block in self.blocks:
            x = block(x)                    # (B, T, d_model)

        x = self.ln_f(x)                    # (B, T, d_model)
        logits = self.lm_head(x)            # (B, T, vocab_size)
        return logits