'''
muliti-head attention
matrix dim:
x: [seq_len, batch_size, d_model]
q, k, v: [seq_len, batch_size, heads, d_k]
qk_score: [seq_len_q, seq_len_k, batch_size, heads]
+mask: [seq_len_q, seq_len_k, batch_size]
softmax: [seq_len_q, seq_len_k, batch_size, heads]
+dropout
mul_v: [seq_len_q, batch_size, heads, d_k]
x: [seq_len_q, batch_size, d_model]
'''

import torch
import torch.nn as nn
import math # for sqrt
from typing import Optional, List
        
class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # if d_model % heads != 0, this linear layer can solve the problem
        self.linear=nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k
        
    def forward(self, x: torch.Tensor):
        # input size: [seq_len, batch_size, d_model] or [batch_size, d_model]
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        # output size: [seq_len, batch_size, heads, d_k]
        return x
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout_prob: float = 0.1, bias: bool = True):
        super.__init__()
        self.d_k = d_model // heads
        self.heads = heads
        # d_model would also be divided to different heads for parallel compution, 
        # rather than being calculated in different heads repeatedly
        # output: [seq_len, batch_size, heads, d_k]
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        self.softmax = nn.Softmax(dim=1) # softmax along the seq_len or time dim
        
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.scale = 1 / math.sqrt(self.d_k)
        
        self.attn = None
        
    def prepare_mask(self, mask:torch.Tensor, query_shape: List[int], key_shape: List[int]):
        # mask size: [seq_len_q, seq_len_k, batch_size]
        # mask is set according to k's positions for each q
        # mask dim0 can be seq_len_q, or just 1 (will be broadcast to seq_len_q then)
        assert mask.shape[0] == 1 or mask.shape[0] ==  query_shape[0]
        # mask dim1 must be same as the key's seq_len
        assert mask.shape[1] == key_shape[0]
        # similar to dim0
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        
        mask = mask.unsqueeze(-1) # add the heads dim [seq_len_q, seq_len_k, batch_size, heads]
        return mask
    
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        # [seq_len, batch_size, heads, d_k] -> [seq_len_q, seq_len_k, batch_size, heads]
        return torch.einsum('ibhd,jbhd->ijbh', query, key)
    
    
    # use * to indicate "keyword-only param", that is to say, 
    # all params after the * cannot be passed as positional args
    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tenor,
                value: torch.Tenor,
                mask: Optional[torch.Tensor] = None):
        # input size: [seq_len, batch_size, d_model] or [batch_size, d_model]
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        # output: [seq_len, batch_size, heads, d_k]
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        scores = self.get_scores(query, key)
        scores *= self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        
        attn = self.softmax(scores)
        
        # tracker.debug('attn', attn)
        
        attn = self.dropout(attn)
        
        x = torch.einsum("ijbh, jbhd->ibhd", attn, value) # multiply by V (ij,jd->id)
        
        self.attn = attn.detach()
        # detach() is a PyTorch method that creates a new tensor that shares the same data but doesn't require gradients
        
        # merge heads and d_k
        x = x.reshape(seq_len, batch_size, -1)
        
        return self.output(x)