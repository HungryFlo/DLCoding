import torch
from torch import nn
from multihead_attention import MultiHeadAttention

def shift_right(x: torch.Tensor):
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    pass
    
class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int, dropout_prob: float=0.1):
        super().__init__(heads, d_model, dropout_prob, bias=False)
        self.P = 2**12
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P*2, heads, self.d_k)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P*2, heads)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.P*2)), requires_grad=True)
        
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0] : self.P + key.shape[0]]
        key_pos_bias = self.key_pos_bias[self.P - key.shape[0] : self.P + key.shape[0]]
        query_pos_bias = self.query_pos_bias[None, None, :, :]
        ac = torch.einsum('ibhd, jbhd->ijbh', query + query_pos_bias, key)
        b = torch.einsum('ibhd, jhd->ijbh', query, key_pos_emb)
        d = key_pos_bias[None, :, None, :]
        bd = shift_right(b+d)
        bd = bd[:, -key.shape[0]:]
        return ac + bd
        