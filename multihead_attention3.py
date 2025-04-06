'''
20250406 mha second time implementation 
Mask need to be added just before softmax.
Key Missing Elements:
    Add the head dim for broadcasting: mask = mask.unsqueeze(-1)
    Dropout layer after softmax: attn = self.dropout(attn) (after softmax and before multiply by V)
    Final output projection layer: return self.output(attention)
    More detailed mask validation: assert mask.dim() == 3, "Mask must be 3D [seq_len_q, seq_len_k, batch_size]"
    Compare the shape, not the value, so you need to use mask^.shape[0]^, rather than mask[0]
'''

import torch
import torch.nn as nn
import math
from typing import Optional, List
from einops import rearrange
from functools import partial

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, head_num: int, head_dim: int, dropout_prob: float = 0.1, qk_bias: bool=True):
        super().__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = head_dim
        self.w_q = nn.Linear(d_model, head_dim*head_num, bias=qk_bias)
        self.w_k = nn.Linear(d_model, head_dim*head_num, bias=qk_bias)
        self.w_v = nn.Linear(d_model, head_dim*head_num, bias=True)
        self.factor = 1.0 / math.sqrt(head_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(head_dim*head_num, d_model)
        self.attn = None
        
    def forward(self, query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor]=None):
        # [seq_len, batch_size, d_model]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # [seq_len, batch_size, head_dim*head_num]
        rearrange_qkv = partial(rearrange, 
                              pattern='seq_len batch_size (head_dim head_num) -> seq_len batch_size head_num head_dim',
                              head_dim=self.head_dim,
                              head_num=self.head_num)
        q, k, v = rearrange_qkv(q), rearrange_qkv(k), rearrange_qkv(v)
        
        # [seq_len, batch_size, head_num, head_dim]
        qk_sim = torch.einsum('ibhd, jbhd -> ijbh', q, k)
        # [seq_len_q, seq_len_k, batch_size, head_num]
        # scale
        qk_sim *= self.factor
        
        # mask validation and application
        if mask is not None:
            assert mask.dim() == 3, "Mask must be 3D [seq_len_q, seq_len_k, batch_size]"
            assert mask.shape[0] == 1 or mask.shape[0] == qk_sim.shape[0]
            assert mask.shape[1] == qk_sim.shape[1]
            assert mask.shape[2] == 1 or mask.shape[2] == qk_sim.shape[2]
            mask = mask.unsqueeze(-1) # [seq_len_q, seq_len_k, batch_size, 1]
            qk_sim = qk_sim.masked_fill(mask==0, float('-inf'))
            
        # attention calculation
        attn = self.softmax(qk_sim)
        attn = self.dropout(attn)
        
        # multiply by V
        attention = torch.einsum('ijbh, jbhd -> ibhd', attn, v)
        self.attn = attention.detach()
        
        # merge heads and project
        attention = rearrange(attention, 'i b h d -> i b (h d)')
        return self.output(attention)        
        