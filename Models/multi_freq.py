import torch
import torch.nn as nn
# from model.module.trans import Transformer as Transformer_s
# from model.module.trans_hypothesis import Transformer
import numpy as np
from einops import rearrange
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn import init
import scipy.sparse as sp

import config

frame_num = config.frame_num




class Multi_freq(nn.Module):
    def __init__(self, d_coor ):
        super(Multi_freq, self).__init__()

        self.d_coor = d_coor
        self.temporal_pos_embedding = nn.Parameter(torch.randn(size=[1,1,300,d_coor],device='cuda'))

        self.proj = nn.Linear(6, d_coor)
        self.proj2 = nn.Linear(d_coor, 6)

        self.stc_block = Multi_freq_Former(self.d_coor)

    def forward(self, input):
        input = rearrange(input,'b c s t -> b s t c')
        input = self.proj(input)
        # blocks layers
        input = input + self.temporal_pos_embedding
        input = self.stc_block(input)
        input = self.proj2(input)
        input = rearrange(input, 'b s t c -> b c s t')
        return input


class Multi_freq_Former(nn.Module):
    def __init__(self, d_coor):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_coor)

        self.mlp = Mlp(d_coor, d_coor, d_coor)

        self.ste_att = Multi_freq_ATTENTION(d_coor)

    def forward(self, input):
        b, s, t, c = input.shape
        x = self.ste_att(input)
        x = x + self.mlp(self.layer_norm(x))

        return x


class Multi_freq_ATTENTION(nn.Module):
    def __init__(self,d_coor, head=8):
        super().__init__()
        # print(d_time, d_joint, d_coor,head)
        self.qkv = nn.Linear(d_coor, d_coor * 3)
        self.q = nn.Linear(d_coor, d_coor)
        self.kv_all = nn.Linear(d_coor,d_coor*2)
        self.kv_half = nn.Linear(d_coor, d_coor * 2)
        self.kv_forth = nn.Linear(d_coor, d_coor * 2)

        self.head = head
        self.layer_norm = nn.LayerNorm(d_coor)
        self.layer_norm_half = nn.LayerNorm(d_coor)
        self.layer_norm_forth = nn.LayerNorm(d_coor)


        self.scale = (d_coor // 2) ** -0.5
        self.proj1 = nn.Linear(d_coor*3, d_coor)

        self.head = head





    def forward(self, input):
        b, s, t, c = input.shape
        h = input
        input_half =  input[:, :, ::2, :] #降采样 得到原来一半的采样 #b 63 150 256
        input_forth = input[:, :, ::4, :]  # 降采样 得到原来4分之1的采样 #b 63 75 256

        x_forth = self.layer_norm_forth(input_forth)
        x_half = self.layer_norm_half(input_half)
        x_all = self.layer_norm(input)

        q_input = self.q(input) #b 63 300 256

        kv_all = self.kv_all(x_all) #b 63 300 256*2
        kv_half = self.kv_half(x_half) #b 63 150 256*2
        kv_forth = self.kv_forth(x_forth)


        kv_all = kv_all.reshape(b, s, t, c, 2).permute(4, 0, 1, 2, 3)  # 2,b,63,300,256
        kv_half = kv_half.reshape(b, s, t//2, c, 2).permute(4, 0, 1, 2, 3)  # 2,b,63,150,256
        kv_forth = kv_forth.reshape(b, s, t // 4, c, 2).permute(4, 0, 1, 2, 3)  # 2,b,63,75,256

        k_all = kv_all[0]
        v_all = kv_all[1]  #b,63,300,256

        k_half = kv_half[0]
        v_half = kv_half[1] #b,63,150,256

        k_forth = kv_forth[0]
        v_forth = kv_forth[1]

        q_input = rearrange(q_input,'b s t (h c) -> (b h s) t c', h=self.head) #(b h 63) 300 c//h
        k_all = rearrange(k_all,'b s t (h c) -> (b h s) c t', h=self.head) #(b h 63) c//h 300
        k_half = rearrange(k_half,'b s t (h c) -> (b h s) c t', h=self.head) #(b h 63) c//h 150
        k_forth = rearrange(k_forth,'b s t (h c) -> (b h s) c t', h=self.head) #(b h 63) c//h 150

        att_all = (q_input @ k_all) * self.scale  # b*h*s,300,300
        att_half = (q_input @ k_half) * self.scale  # b*h*s,300,150
        att_forth = (q_input @ k_forth) * self.scale  # b*h*s,300,150

        att_all = att_all.softmax(-1)  # # b*h*s,300,300
        att_half = att_half.softmax(-1)  # b*h*s,300,150
        att_forth = att_forth.softmax(-1)

        v_all = rearrange(v_all, 'b s t (h c) -> (b h s) t c',h=self.head)
        v_half = rearrange(v_half, 'b s t (h c) -> (b h s) t c',h=self.head)
        v_forth = rearrange(v_forth, 'b s t (h c) -> (b h s) t c', h=self.head)

        x_all = att_all @ v_all    # b*h*t,s,c//2//h
        x_half = att_half @ v_half  # b*h,t,c//h                # b*h*s,t,c//2//h
        x_forth = att_forth @ v_forth

        x_all = rearrange(x_all, '(b h s) t c -> b s t (h c)', h=self.head, b=b)  # b*h*t,s,c//h//2 -> b,h,t,s,c//h//2
        x_half = rearrange(x_half, '(b h s) t c -> b s t (h c)', h=self.head, b=b)  # b*h*s,t,c//h//2 -> b,h,t,s,c//h//2
        x_forth = rearrange(x_forth, '(b h s) t c -> b s t (h c)', h=self.head, b=b)  # b*h*s,t,c//h//2 -> b,h,t,s,c//h//2
        
        output = torch.concat((x_all,x_half,x_forth),-1)
        output = self.proj1(output)
        output = output + h
        return output




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





if __name__ == "__main__":
    # inputs = torch.rand(64, 351, 34)  # [btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    multi_freq = Multi_freq(d_coor=256)
    inputs = torch.rand([3,6, 63, 300])
    output = net(inputs)
    print(output.size())

