'''
get_positional_encoding

'''

import torch
import torch.nn as nn
import math
from copy import deepcopy

from multihead_attention import MultiHeadAttention


def get_positional_encoding(d_model: int, max_len: int=5000):
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # 0::2 means select every 2nd column starting from index 0
    # : means select all rows
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    
    return encodings

def clone_module_list(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(n)])

class EmbeddingsWithPositionalEncoding(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_vocab: int,
                 max_len: int=5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))
        """ 
        Key points about register_buffer():
            It registers a tensor that should be part of the module's state but isn't a parameter
            Buffers are saved when calling state_dict() and restored when calling load_state_dict()
            They are typically used for:
            Positional encodings
            Running statistics in BatchNorm
            Any persistent state that isn't a learnable parameter
            The registered buffer can be accessed as an attribute of the module
            Buffers are moved to the same device as the module when using .to(device) 
        """
        
    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        # multiply by sqrt(d_model) to scales the embeddings 
        # to have a similar magnitude as the positional encodings
        return self.linear(x) * math.sqrt(self.d_model) + pe
    
class EmbeddingsWithLearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int,
                 n_vocab: int,
                 max_len: int=5000):
        super().__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(
            torch.zeros(max_len, 1, d_model),
            requires_grad=True
        )
    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe
    
class TransformerLayer(nn.Module):
    '''
    can act as an encoder layer or a decoder layer
    use pre-norm
    '''
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention, # Self Attn module
                 src_attn: MultiHeadAttention = None, # Source Attention module (decoder),
                 feed_forward: FeedForward,
                 dropout_prob: float):
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        # nn.LayerNorm(normalized_shape, ...)
        # normalized_shape indicates the dim list of you want to normalize, 
        # which must be last certain dims
        self.norm_src_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        
        self.is_save_ff_input = False
        
        
    def forward(self, *,
                x: torch.Tensor,
                mask: torch.Tensor,
                src: torch.Tensor=None,
                src_mask: torch,Tensor=None):
        # norm before self_attn
        z = self.norm_self_attn(x)
        # self_attn
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        # res link
        x = x + self.dropout(self_attn)
        
        # cross_attn in decoder block
        if src is not None:
            z = self.norm_src_attn(x)
            # query from decoder, key and value from encoder
            attn_src = self.src_attn(query=z, key=src, value=src, mask = src_mask)
            x = x + self.dropout(attn_src)
        # norm before feed-forward
        z = self.norm_ff(x)
        
        # save the input to ff layer if specified
        if self.is_save_ff_input:
            self.ff_input = z.clone()
        
        # ff layer
        ff = self.feed_forward(z)
        # res link
        x = x + self.dropout(ff)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, layer: TransformerLayer, n_layers: int):
        super().__init__()
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])
    
    def forward(self, x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        return self.norm(x)
    
class Generator(nn.Module):
    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)
        
    def forward(self, x):
        return self.projection(x)
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder,
                 decoder: Decoder,
                 src_embed: nn.Module,
                 tgt_embed: nn.Module,
                 generator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
        for p in self.parameters():
            if p.dim() > 1:
                # initialize the params using Xavier method
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    
    def forward(self, src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        enc = self.encode(src, src_mask)
        return self.decode(enc, src_mask, tgt, tgt_mask)