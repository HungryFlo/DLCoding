'''
calculation trick:
target: 
[q_0 q_1 q_2 q_3 ... q_{d-2} q_{d-1}] .* [cos(m*theta_0) cos(m*theta_0) cos(m*theta_1) cos(m*theta_1) ... cos(m*theta_{2/d - 1}) cos(m*theta_{2/d - 1})] + 
[-q_1 q_0 -q_3 q_2 ... -q_{d-1} q_{d-2}] .* [sin(m*theta_0) sin(m*theta_0) sin(m*theta_1) sin(m*theta_1) ... sin(m*theta_{2/d - 1}) sin(m*theta_{2/d - 1})]

step1: [m*theta_0 m*theta_1 ... m*theta_{d/2}] --self-concat--> [m*theta_0 m*theta_1 ... m*theta_{d/2} m*theta_0 m*theta_1 ... m*theta_{d/2}]
step2: [q_0 q_1 q_2 q_3 ... q_{d-2} q_{d-1}] --half-divide-concat--> [-q_{d//2 + 1} -q_{d//2 +2} ... -q_{d-1}  q_0 q_1 q_2 q_3 ... q_{d//2-1} q_{d//2}]
step3:  [q_0            q_1         ...   q_{d//2}     q_{d//2 + 1} q_{d//2 +2}  ...       q_{d-1}]
        [-q_{d//2 + 1} -q_{d//2 +2} ...  -q_{d-1}      q_0          q_1          ...       q_{d//2}]
        [m*theta_0      m*theta_1   ...  m*theta_{d/2} m*theta_0    m*theta_1    ...       m*theta_{d/2}]
'''

import torch
from torch import nn
from multihead_attention import MultiHeadAttention

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int=10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None
        
    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[0]
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        # [seq_len, batch_size, n_heads, d]
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]
        
    def _neg_half(self, x:torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[:,:,:,d_2:], x[:,:,:,:d_2]], dim=-1)
    
    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return torch.cat((x_rope, x_pass), dim=-1)
    
class RotaryPEMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model: int, heads: int, rope_percentage: float=0.5, dropout_prob: float=0.1, bias: bool=True):
        super().__init__(heads=heads, d_model=d_model, dropout_prob=dropout_prob, bias=bias)
        d_rope = int(self.d_k * rope_percentage)
        self.query_rotary_pe = RotaryPositionalEmbeddings(d_rope)
        self.key_rotary_pe = RotaryPositionalEmbeddings(d_rope)
    def get_scores(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        return torch.einsum('ibhd, jbhd->ijbh', self.query_rotary_pe(query), self.key_rotary_pe(key))
    
        
        
        