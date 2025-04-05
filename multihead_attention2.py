import torch
import torch.nn as nn
import math
from typing import Optional, List
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, model_dim, dropout_prob: float = 0.1, qk_bias: bool =True):
        super().__init__()
        self.head_num = head_num
        self.model_dim = model_dim
        self.head_dim = model_dim // head_num
        # self.query = self.prepare_qkv(bias=qk_bias)
        # self.key = self.prepare_qkv(bias=qk_bias)
        # self.value = self.prepare_qkv(bias=True)
        self.w_q = nn.Linear(model_dim, self.head_num*self.head_dim, bias=qk_bias)
        self.w_k = nn.Linear(model_dim, self.head_num*self.head_dim, bias=qk_bias)
        self.w_v = nn.Linear(model_dim, self.head_num*self.head_dim, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = None
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor]=None):
        # q,k,v: [seq_len, batch_size, model_dim]
        # get q,k,v
        # q,k,v use different matrix
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)
        query = rearrange(query, 'seq bs (head_num head_dim) -> seq bs head_num head_dim', 
                      head_num=self.head_num, head_dim=self.head_dim)
        key = rearrange(key, 'seq bs (head_num head_dim) -> seq bs head_num head_dim', 
                      head_num=self.head_num, head_dim=self.head_dim)
        value = rearrange(value, 'seq bs (head_num head_dim) -> seq bs head_num head_dim', 
                      head_num=self.head_num, head_dim=self.head_dim)
        # calculate qkT / sqrt(d_head)
        q_k_sim = torch.einsum('ibhd,jbhd->ijbh', query, key)
        q_k_sim *= 1 / math.sqrt(self.head_dim)
        # mask: [seq_len_q, seq_len_k, batch_size]
        assert mask.shape[0] == 1 or mask.shape[0] == q_k_sim.shape[0]
        assert mask.shape[1] == q_k_sim.shape[1]
        assert mask.shape[2] == 1 or mask.shape[2] == q_k_sim.shape[2]
        if mask is not None:
            q_k_sim = q_k_sim.masked_fill(mask==0, float('-inf'))
        # softmax on seq_len_k dim (dim=1)
        q_k_sim = self.softmax(q_k_sim)
        # multiply by V
        attention = torch.einsum('ijbh,jbhd->ibhd', q_k_sim, value)
        # dropout
        attention = self.dropout(attention)
        # save attention result
        self.attention = attention.detach()
        # reshape and return
        attention = rearrange(attention, 'i b h d -> i b (h d)')
        return attention