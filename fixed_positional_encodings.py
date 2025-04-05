import math

import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int=5000):
        super.__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)
    
    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = x + pe
        x = self.dropout(x)
        return x
        
def get_positional_encoding(d_model: int, max_len: int=5000):
    encodings = torch.zeros(max_len, d_model)
    position = torch.arrange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arrange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # 0::2 means select every 2nd column starting from index 0
    # : means select all rows
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    
    return encodings
        
        
def get_positional_encoding2(d_model: int, max_len: int=5000):
    encodings = torch.zeros(max_len, d_model)
    # positions = torch.arrange(0, max_len).unsqueeze(1)
    positions = torch.arrange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arrange(0, d_model, 2, dtype=torch.float32)
    # factor = torch.exp(two_i * -(^^np^^.log(^^10000^^) / d_model))
    factor = torch.exp(two_i * -(np.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(positions * factor)
    encodings[:, 1::2] = torch.cos(positions * factor)
    
    # encodings.^^require^^_grad_(False)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    
    return encodings