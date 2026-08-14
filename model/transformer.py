import torch.nn as nn 
from model.sca import SCA
from model.block import TransformerBlock
from model.norms import RMSNorm

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.token_emb(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_kv_head, n_layers, context_length,
                 n_head_memory, n_head_query, spectral_sample, head_dim,
                 sca_ratio=1, d_ff=None, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            self._build_layers(sca_ratio, d_model, n_head, n_kv_head, context_length,
                               n_head_memory, n_head_query, spectral_sample, head_dim,
                               d_ff, dropout)
        )
        self.ln_f = RMSNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def _build_layers(self, sca_ratio, d_model, n_head, n_kv_head, context_length,
                      n_head_memory, n_head_query, spectral_sample, head_dim, d_ff, dropout):
        layers = []
        for _ in range(self.n_layers):
            for _ in range(sca_ratio):
                layers.append(SCA(d_model, n_head_memory, n_head_query,
                                  spectral_sample, head_dim))
            layers.append(TransformerBlock(d_model, n_head, n_kv_head,
                                           context_length, d_ff, dropout))
        return layers

    def forward(self, x, states=None):
        """states : un état par couche, dans l'ordre des couches, ou None.

        SCA et TransformerBlock partagent la signature (x, state) -> (out, state),
        d'où l'absence de dispatch par type. Les états sont toujours renvoyés :
        c'est le prefill du prompt qui les produit pour la génération.
        """
        x = self.token_emb(x)
        x = self.dropout(x)

        new_states = []
        for k, layer in enumerate(self.layers):
            x, new_state = layer(x, None if states is None else states[k])
            new_states.append(new_state)

        x = self.ln_f(x)
        x = self.output_layer(x)
        return x, new_states
