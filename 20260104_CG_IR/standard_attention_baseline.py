import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class StandardAttention(nn.Module):
    """Standard attention with fixed temperature for baseline comparison"""

    def __init__(self, dim, num_heads, bias, fixed_temperature=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.fixed_temperature = fixed_temperature

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.fixed_temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        out = self.project_out(out)

        return out, attn

def create_baseline_model(adaptive_model):
    """Create a baseline model with standard attention for comparison"""
    baseline_model = type(adaptive_model).__new__(type(adaptive_model))
    baseline_model.__dict__.update(adaptive_model.__dict__)
    return baseline_model
