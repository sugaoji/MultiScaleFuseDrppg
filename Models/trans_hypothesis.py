import math
import torch
import torch.nn as nn
from functools import partial

from einops import rearrange
from timm.models.layers import DropPath

import config


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = Mlp(dim,dim,dim)
        self.pos_embed_1 = nn.Parameter(torch.randn(1, 300,1, dim))
        self.pos_embed_2 = nn.Parameter(torch.randn(1, 300,1, dim))
        self.pos_embed_3 = nn.Parameter(torch.randn(1, 300,1, dim))
    def forward(self, x_1, x_2, x_3):
        x_1 = x_1 + self.pos_embed_1
        x_2 = x_2 + self.pos_embed_2
        x_3 = x_3 + self.pos_embed_3

        B, T, N, C = x_1.shape
        q = self.linear_q(x_1)
        k = self.linear_k(x_2)
        v = self.linear_v(x_3)

        q = rearrange(q, 'B T N (H C) -> (B H N) T C', H=self.num_heads)
        k = rearrange(k, 'B T N (H C) -> (B H N) T C', H=self.num_heads)
        v = rearrange(v, 'B T N (H C) -> (B H N) T C', H=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v)
        x = rearrange(x, '(B H N) T C -> B T N (H C)', B=B, H=self.num_heads)
        x = self.proj(x)
        return x


class CHI_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, drop=0.,
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm3_11 = norm_layer(dim)
        self.norm3_12 = norm_layer(dim)
        self.norm3_13 = norm_layer(dim)

        self.norm3_21 = norm_layer(dim)
        self.norm3_22 = norm_layer(dim)
        self.norm3_23 = norm_layer(dim)

        self.norm3_31 = norm_layer(dim)
        self.norm3_32 = norm_layer(dim)
        self.norm3_33 = norm_layer(dim)

        self.attn_1 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop)
        self.attn_2 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop)
        self.attn_3 = Cross_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop)


        self.norm2 = norm_layer(dim * 3)
        self.mlp = Mlp(in_features=dim * 3, hidden_features=dim * 3, act_layer=act_layer, drop=drop)

    def forward(self, x_1, x_2, x_3):
        x_1 = x_1 + self.attn_1(self.norm3_11(x_2), self.norm3_12(x_3), self.norm3_13(x_1))
        x_2 = x_2 + self.attn_2(self.norm3_21(x_3), self.norm3_22(x_1), self.norm3_23(x_2))
        x_3 = x_3 + self.attn_3(self.norm3_31(x_1), self.norm3_32(x_2), self.norm3_33(x_3))

        x = torch.cat([x_1, x_2, x_3], dim=3)
        x = x + self.mlp(self.norm2(x))

        x_1 = x[:, :, :, :x.shape[3] // 3]
        x_2 = x[:, :, :, x.shape[3] // 3: x.shape[3] // 3 * 2]
        x_3 = x[:, :, :, x.shape[3] // 3 * 2: x.shape[3] // 3 * 3]

        return x_1, x_2, x_3


class Transformer(nn.Module):
    def __init__(self,embed_dim=256, mlp_hidden_dim=256, h=8,
                 drop_rate=0.1):
        super().__init__()



        self.CHI_block = CHI_Block(
            dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim,
            drop=drop_rate, act_layer=nn.GELU)

        # self.regression = Mlp(embed_dim*3,embed_dim,embed_dim)

        self.regression = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x_1, x_2, x_3):
        # x_1 = self.pos_embed_1 + x_1
        # x_2 = self.pos_embed_2 + x_2
        # x_3 = self.pos_embed_3 + x_3
        #
        # for i, blk in enumerate(self.SHR_blocks):
        #     x_1, x_2, x_3 = self.SHR_blocks[i](x_1, x_2, x_3)


        x_1, x_2, x_3 = self.CHI_block(x_1, x_2, x_3)

        x = torch.cat([x_1, x_2, x_3], dim=3)
        x = self.regression(x)  # b 63 256 300

        # x = torch.mean(torch.stack([x_1, x_2, x_3],dim=0), dim=0)


        return x
#现在这个版本是把cross attention在时间T维度上做，以前是在ROI维度上做的
#把最后经过cross attention 得到的3个特征直接加起来，而不是concat
#以上两个都改回来，但是3个特征改为1 3 5 而不是最后3个


