import math
import torch

from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt
import random

import numpy as np
import math
from scipy import signal

import config
from Models import trans_hypothesis

# class FreqMlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         b, f, _ = x.shape
#         x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
#         return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args):
        if len(args) == 0:
            return self.fn(self.norm(x))
        else:
            return self.fn(self.norm(x), self.norm(args[0]))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SpatialTransformer(nn.Module):
    def __init__(self, dim, heads=4, dim_head=4, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.spatial_attention = Attention(dim, heads, dim_head, dropout)
        self.dropout = nn.Dropout(dropout)
        #在这里添加poseformerv2代码

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        spatial_attention, v = self.spatial_attention(x)
        spatial_attention = self.dropout(spatial_attention)
        out = torch.matmul(spatial_attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class TemporalTransformer(nn.Module):
    def __init__(self, dim, heads=4, dim_head=4, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.temporal_attention = Attention(dim, heads, dim_head, dropout)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        temporal_attention, v = self.temporal_attention(x)
        temporal_attention = self.dropout(temporal_attention)
        out = torch.matmul(temporal_attention, v)
        out = rearrange(out, 'b h t d -> b t (h d)')

        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=4, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        # attn = self.dropout(attn)

        # out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return attn, v






class SpatialTemporalInteraction(nn.Module):
    def __init__(self, dim, num_clusters, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.num_clusters = num_clusters
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SpatialTransformer(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm(dim, TemporalTransformer(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.pos_embedding = nn.Parameter(torch.randn(1, 300, dim))

    def forward(self, x):
        count = 0
        for spatial_attn, ff1, temporal_att, ff2 in self.layers:
            num_clusters = x.shape[-2]
            x = spatial_attn(x) + x
            x = ff1(x) + x
            x = rearrange(x, '(B T) K D -> (B K) T D', T=300)
            if count == 0:
                x += self.pos_embedding[:, :300]
                x = self.dropout(x)
            x = temporal_att(x) + x
            x = ff2(x) + x
            x = rearrange(x, '(B K) T D -> (B T) K D', K=self.num_clusters)
            count = count + 1

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, dim, num_patch, emb_dropout):
        super().__init__()

        # patch_dim = channels
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b d1 d2 c -> (b d1) d2 c'),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, stmap):
        x = self.to_patch_embedding(stmap)
        b, num_patch, _ = x.shape

        x += self.pos_embedding[:, :num_patch]
        x = self.dropout(x)

        return x


class Embedding(nn.Module):
    def __init__(self, dim, num_clusters, emb_dropout, channels):
        super().__init__()

        self.num_clusters = num_clusters

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, dim),  # dim = 64
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, stmap):
        stmap = stmap.to(torch.float32)
        x = self.to_patch_embedding(stmap)
        x += self.pos_embedding[:, :self.num_clusters]
        x = self.dropout(x)

        return x










if __name__ == "__main__":
    v = ViT(
        image_height=63,
        image_width=300,
        num_classes=300,
        num_clusters=6,
        dim=16,
        depth=10,
        heads=4,
        mlp_dim=16,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(2, 3, 63, 300)
