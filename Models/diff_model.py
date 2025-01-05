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
from scipy import signal
from Models.mix_ste import *
import config

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def extract(a, t, x_shape):
    """extract the appropriate t index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

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




class Embedding(nn.Module):
    def __init__(self, dim, num_clusters, emb_dropout, channels):
        super().__init__()
        self.num_clusters = num_clusters
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, dim), 
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

class mixste(nn.Module):
    def __init__(self, num_clusters, dim, mlp_dim, depth, heads, dim_head, dropout, emb_dropout, channels):
        super().__init__()
        C = channels
        self.input_embedding = Embedding(dim, num_clusters, emb_dropout, C)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.spatial_temporal_interaction = stcModel(layers=depth, d_hid=dim, frames=300, n_joints=num_clusters, out_joints=num_clusters)

    def forward(self, noise_bvps, mstmaps, time_cond):
        batch = mstmaps.shape[0]
        reshaped_bvps = noise_bvps.view(batch, 1, 1, 300)
        repeat_bvps = reshaped_bvps.expand(batch, 1, 63, 300)
        mstmaps_plus_diffused_bvps = torch.cat((mstmaps, repeat_bvps), dim=1)
        B, C, N, T = mstmaps_plus_diffused_bvps.shape
        input = rearrange(mstmaps_plus_diffused_bvps, 'B C N T -> (B T) N C')
        input = self.input_embedding(input)
        input = rearrange(input, '(B T) N C -> B T N C', T=300)
        time_embed = self.time_mlp(time_cond)[:, None, None, :]
        input = input + time_embed
        x_spatial = self.spatial_temporal_interaction(input)
        return x_spatial

class ViT(nn.Module):
    def __init__(self, image_height, image_width, num_classes, num_clusters, dim, mlp_dim, depth, heads, channels=7,
                 dim_head=4, dropout=0., emb_dropout=0., is_test=False, sampling_timesteps=config.K):
        super().__init__()
        dim_head = dim // heads
        N = image_height
        T = image_width
        C = channels
        self.is_test = is_test
        self.scale = 0.5
        self.num_timesteps = config.Total_steps
        sampling_timesteps = sampling_timesteps
        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.num_clusters = num_clusters
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        self.pred = mixste(num_clusters, dim, mlp_dim, depth, heads, dim_head, dropout, emb_dropout, channels)
        self.device = 'cuda'

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, noise_bvps, mstmaps, t):
        x_t = noise_bvps
        x_t = torch.clamp(x_t, min=-2.2 * self.scale, max=2.2 * self.scale)
        x_t = x_t / self.scale
        pred_bvps = self.pred(x_t, mstmaps, t)
        pred_bvps = (pred_bvps - torch.mean(pred_bvps)) / torch.std(pred_bvps)
        x_start = pred_bvps
        x_start = x_start * self.scale
        x_start = torch.clamp(x_start, min=-2.2 * self.scale, max=2.2 * self.scale)
        pred_noise = self.predict_noise_from_start(noise_bvps, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, mstmaps):
        predict_list = []
        batch = mstmaps.shape[0]
        shape = (batch, 300)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        noise_bvps = torch.randn(shape, device=self.device)
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            preds = self.model_predictions(noise_bvps, mstmaps, time_cond)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            if time_next < 0:
                noise_bvps = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(noise_bvps)
            noise_bvps = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return noise_bvps

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def diffuse_bvp(self, bvps):
        diffused_bvps = []
        ts = []
        for i in range(0, bvps.shape[0]):
            bvp = bvps[i]
            t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
            noise = torch.randn(300, device='cuda')
            x_start = bvp
            x_start = x_start * self.scale
            x = self.q_sample(x_start=x_start, t=t, noise=noise)
            x = torch.clamp(x, min=-2.2 * self.scale, max=2.2 * self.scale)
            x = x / self.scale
            d_poses, d_noise, d_t = x, noise, t
            diffused_bvps.append(d_poses)
            ts.append(d_t)
        return torch.stack(diffused_bvps), torch.stack(ts)

    def forward(self, mstmaps, bvps):
        if not self.is_test:
            diffused_bvps, ts = self.diffuse_bvp(bvps)
            diffused_bvps = diffused_bvps.float()
            ts = ts.squeeze(-1)
            pred_results = self.pred(diffused_bvps, mstmaps, ts)
            return pred_results
        elif self.is_test:
            results = self.ddim_sample(mstmaps)
            return results

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