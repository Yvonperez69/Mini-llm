import torch
import torch.nn as nn
import torch.nn.functional as F
from model.norms import RMSNorm

class SCA(nn.Module):
    def __init__(self, d_model, n_head_memory, n_head_query, spectral_sample, head_dim):
        super().__init__()
        assert n_head_memory == n_head_query
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_head_memory = n_head_memory
        self.n_head_query = n_head_query # il faut query == memory
        self.spectral_sample = spectral_sample

        self.gamma = nn.Parameter(torch.ones(n_head_memory))
        self.beta = nn.Parameter(torch.zeros(n_head_memory))
        self.log_lambda = nn.Parameter(torch.zeros(n_head_memory)) # dans le forward : lambda = F.softplus(self.log_lambda)
        self.eta = nn.Parameter(torch.ones(n_head_memory))
        self.theta = nn.Parameter(torch.ones(n_head_memory, head_dim, spectral_sample))
        self.omega = nn.Parameter(torch.ones(n_head_query, head_dim, spectral_sample))

        self.w_in = nn.Linear(d_model,n_head_memory*(1+head_dim) + 2*n_head_query*head_dim*spectral_sample)
        self.w_gate = nn.Linear(d_model, 2 * n_head_query * head_dim)
        self.w_read = nn.Linear(2*n_head_query*head_dim, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, bias=False)
        self.rms_norm = RMSNorm(2*n_head_query*head_dim)

    def forward(self, x, state = None):
        """state = (R, I, Z, conv) ou None au début d'une séquence.

        (R, I, Z) portent les sommes préfixes de l'étape 4, conv porte la fenêtre
        glissante de la conv causale de l'étape 1. Les deux mémoires doivent
        traverser la frontière entre deux appels pour que la génération
        token par token donne les mêmes valeurs que la séquence entière.
        """
        B, T, C = x.shape #batch size, sequence length, embedding dimension
        R0 = I0 = Z0 = conv_state = None
        if state is not None:
            R0, I0, Z0, conv_state = state

        # step 1
        z = x.transpose(1, 2)
        pad = self.dw_conv.kernel_size[0] - 1
        # Padding de zéros seulement en début de séquence : là, "avant" est
        # réellement vide. En cours de génération, "avant" ce sont les `pad`
        # derniers pas de temps — sinon la conv perd son champ réceptif.
        if conv_state is None:
            z = F.pad(z, (pad, 0))
        else:
            z = torch.cat([conv_state, z], dim=-1)
        new_conv_state = z[:, :, -pad:]       # fenêtre pour le prochain appel
        z = self.dw_conv(z).transpose(1, 2)  # conv sur d_model channels
        z = self.w_in(z)                      # projection après
        z = F.silu(z)
        zmem, zquery = torch.split(z, [self.n_head_memory*(1+self.head_dim), 2*self.n_head_query*self.head_dim*self.spectral_sample], dim=-1)
        k, s = torch.split(zmem, [self.n_head_memory*self.head_dim, self.n_head_memory], dim=-1)
        k = k.view(B, T, self.n_head_memory, self.head_dim)
        s = s.view(B, T, self.n_head_memory)
        zquery = zquery.view(B, T, self.n_head_query, self.head_dim, 2, self.spectral_sample)
        q_re = zquery[..., 0,:]
        q_im = zquery[..., 1,:]

        # step 2
        d = 0 # distance-to-boundary function , pour l'instant on laisse à 0
        lam = F.softplus(self.log_lambda)
        alpha = F.softplus(self.gamma*s + self.beta)*torch.exp(-lam*d)

        # step 3
        eta = self.eta.unsqueeze(-1)
        phi = F.softsign(eta*k).unsqueeze(-1)* self.theta
        r = alpha.unsqueeze(-1).unsqueeze(-1)*k.unsqueeze(-1)*torch.cos(phi)
        i = alpha.unsqueeze(-1).unsqueeze(-1)*k.unsqueeze(-1)*torch.sin(phi)

        # step 4
        # Une seule branche : cumsum donne les sommes préfixes à l'intérieur du
        # chunk, l'état ajoute tout ce qui précède. Pour T=1 le cumsum est
        # l'identité, donc le mode récurrent en est un cas particulier.
        R = torch.cumsum(r, dim=1)
        I = torch.cumsum(i, dim=1)
        Z = torch.cumsum(alpha.unsqueeze(-1).unsqueeze(-1), dim=1)

        if R0 is not None:
            R = R + R0
            I = I + I0
            Z = Z + Z0

        R_hat = R/(Z + 1e-8)
        I_hat = I/(Z + 1e-8)

        # Le total courant, toujours de longueur 1 quel que soit T.
        # detach : à l'entraînement l'état n'est pas réutilisé, inutile de
        # retenir le graphe autograd.
        new_state = (R[:, -1:].detach(), I[:, -1:].detach(),
                     Z[:, -1:].detach(), new_conv_state.detach())

        # step 5
        omega = self.omega.unsqueeze(0).unsqueeze(0) # (1, 1, K', H, M)
        o_re = torch.sum(omega*(R_hat*q_re + I_hat*q_im), dim=-1)
        o_im = torch.sum(omega*(I_hat*q_re - R_hat*q_im), dim=-1)

        # step 6
            #GatedRMSNorm
        o = torch.cat([o_re, o_im], dim=-1)  # (B, T, K', H*2) → à reshaper en (B, T, 2*K'*H)
        o = o.reshape(B, T, self.n_head_query * self.head_dim * 2)
        gate = torch.sigmoid(self.w_gate(x))
        o = self.rms_norm(o) * gate
            #Projection + SwiGLU
        o = self.w_read(o)
        o = F.silu(o)*o
            #Projection final
        o = self.w_out(o)

        return o + x, new_state #résidu
