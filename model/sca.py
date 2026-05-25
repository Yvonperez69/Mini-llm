"""
SeqCond Attention (SCA) — couche spectrale hybride.

Implémentation basée sur :
    Nautile-370M: Spectral Memory Meets Attention in a Small Reasoning Model
    Chenebaux & al., 2026 (arXiv:2604.24809)

Forward pass (Algorithm 1 du papier) :
    1. Projection Win + DWConv causal + SiLU  → zmem, zquery
    2. Contribution weight α = softplus(γs + β) × exp(−λ·d(t))
    3. Phase ϕ = softsign(η·k) ⊙ θ,  encodage complexe r + i·i
    4. Cumsum causal sur r, i, α  → état normalisé R̂, Î
    5. Readout hermitien  → o_re, o_im
    6. GatedRMSNorm + SwiGLU + projection → sortie D
"""

