import torch
import torch.nn as nn
import math
from copy import deepcopy
from typing import Optional, Tuple

from multihead_attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    """位置编码实现"""
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer('positional_encoding', get_positional_encoding(d_model,max_len), False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [seq_len, batch_size, d_model]
        return x * math.sqrt(self.d_model) + self.positonal_encoding[:x.shape[0]]
        
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Parameter(
            # first param: initialized tensor, be careful with torch size
            torch.zeros([max_len, 1, d_model]), # [seq_len, batch_size, d_model]
            # second param: requires_grad
            requires_grad=True
        )
    def forward(self, x: torch.Tensor):
        return x * math.sqrt(self.d_model) + self.positional_encoding[:x.shape[0]]

class Embeddings(nn.Module):
    """词嵌入层"""
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [seq_len, batch_size, vocab_size]
        # output: [seq_len, batch_size, d_model]
        return self.embedding(x)
        

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # bias is default to be True
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [seq_len, batch_size, d_model]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class TransformerLayer(nn.Module):
    """
    Encoder Layer: mha + ff
    Decoder Layer: mha + src_mha + ff
    """
    def __init__(self,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: Optional[MultiHeadAttention],
                 feed_forward: FeedForward,
                 dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.norm = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = feed_forward
        
    
    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                src: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input: x[seq_len, batch_size, d_model]
        sa = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = self.norm(self.dropout(sa) + x)
        
        if src is not None:
            # query from decoder, k,v from encoder
            ca = self.src_attn(query=x, key=src, value=src, mask=src_mask)
            x = self.norm(self.dropout(ca) + x)
        
        ff = self.feed_forward(x)
        x = self.norm(self.dropout(ff) + x)
        return x
        

class Encoder(nn.Module):
    """编码器实现"""
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    

class Decoder(nn.Module):
    """编码器实现"""
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
    
    def forward(self, x: torch.Tensor, tgt_mask: torch.Tensor, src: torch.Tensor, src_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, tgt_mask, src, src_mask)
        return x

class Generator:
    """生成器(输出层)"""
    pass

class EncoderDecoder:
    """完整的Transformer模型"""
    pass

# 辅助函数
def get_positional_encoding(d_model: int, max_len: int = 5000) -> torch.Tensor:
    """生成位置编码"""
    pass

def clone_module_list(module: nn.Module, n: int) -> nn.ModuleList:
    """克隆模块列表"""
    pass
