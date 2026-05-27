import torch
import torch.nn as nn
import torch.nn.functional as F 
from norms import RMSNorm

class SCA(nn.Module):
    def __init__(self, d_model, n_head_memory, n_head_query, spectral_sample, head_dim):
        super().__init__()
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
        
    def forward(self, x):
        B, T, C = x.shape #batch size, sequence length, embedding dimension
        # step 1
        z = x.transpose(1, 2)
        z = F.pad(z, (self.dw_conv.kernel_size[0] - 1, 0))
        z = self.dw_conv(z).transpose(1, 2)  # conv sur d_model channels
        z = self.w_in(z)                      # projection après
        z = F.silu(z)
        zmem, zquery = torch.split(z, [self.n_head_memory*(1+self.head_dim), 2*self.n_head_query*self.head_dim*self.spectral_sample], dim=-1)
        k, s = torch.split(zmem, [self.n_head_memory*self.head_dim, self.n_head_memory], dim=-1)
        k = k.view(B, T, self.n_head_memory, self.head_dim)
        s = s.view(B, T, self.n_head_memory)
        zquery = zquery.view(B, T, self.n_head_query, self.head_dim, self.spectral_sample, 2)
        q_re = zquery[..., 0]
        q_im = zquery[..., 1]
        
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
        R = torch.cumsum(r,dim=1)
        I = torch.cumsum(i,dim=1)
        Z = torch.cumsum(alpha.unsqueeze(-1).unsqueeze(-1),dim=1)
        R_hat = R/(Z + 1e-8)
        I_hat = I/(Z + 1e-8)
        
        # step 5
        o_re = torch.sum(self.omega*(R_hat*q_re + I_hat*q_im),dim=-1)
        o_im = torch.sum(self.omega*(I_hat*q_re - R_hat*q_im),dim=-1)
        
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
        
        return o + x #résidu


sca = SCA(d_model=512, n_head_memory=8, n_head_query=8, spectral_sample=2, head_dim=64)
x = torch.randn(2, 16, 512)
out = sca(x)
print(out.shape)  # doit donner torch.Size([2, 16, 512])

# Test 1 — shape générale
sca = SCA(d_model=512, n_head_memory=8, n_head_query=8, spectral_sample=2, head_dim=64)
x = torch.randn(2, 16, 512)
out = sca(x)
assert out.shape == x.shape, "shape incorrecte"
print("✅ Test shape OK")

# Test 2 — causalité
x = torch.randn(1, 10, 512)
out1 = sca(x)
x2 = x.clone()
x2[:, 5:, :] = torch.randn(1, 5, 512)  # on modifie le futur
out2 = sca(x2)
assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5), "causalité brisée"
print("✅ Test causalité OK")

# Test 3 — gradient
x = torch.randn(2, 16, 512, requires_grad=True)
out = sca(x)
loss = out.sum()
loss.backward()
assert x.grad is not None, "pas de gradient sur x"
for name, p in sca.named_parameters():
    assert p.grad is not None, f"pas de gradient sur {name}"
print("✅ Test gradient OK")