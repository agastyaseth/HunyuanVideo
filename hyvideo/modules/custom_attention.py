import torch
import torch.nn as nn

from .attenion import attention


class CustomSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, tracker_name="self_attn_layer"):
        super().__init__()
        self.tracker_name = tracker_name
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, **kwargs):
        # Use the vanilla mode for simplicity, pass batch_size and the tracker name to store attention maps
        batch_size = query.size(0)
        attn_output = attention(query, key, value, mode="vanilla", batch_size=batch_size, tracker_name=self.tracker_name, **kwargs)
        attn_output = self.out_proj(attn_output)
        return attn_output


class CustomCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, tracker_name="cross_attn_layer"):
        super().__init__()
        self.tracker_name = tracker_name
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, **kwargs):
        batch_size = query.size(0)
        attn_output = attention(query, key, value, mode="vanilla", batch_size=batch_size, tracker_name=self.tracker_name, **kwargs)
        attn_output = self.out_proj(attn_output)
        return attn_output 