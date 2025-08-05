import math

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import nn
import random


# absolute positional embedding used for vanilla transformer sequential data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


# very similar positional embedding used for diffusion timesteps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# dropout mask
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def trajectory_masking(x, mask_rates, max_mask_len = 15, joint_mask = 5):
    x_using = x.clone()
    T = x_using.size(1)
    data_dim = x_using.size(-1)
    
    mask = torch.ones_like(x_using[:, :, :, 0])
    mask_joints = None
    rand_number = random.random()
    
    if rand_number < 0.1: # no masking
        return x_using
    
    if joint_mask is not None and rand_number < 0.8: # mask joints / spatial masking - 5 joints
        mask_joints = random.sample([0,1,2,3,4,5], 5)
        mask[:, :, mask_joints] *= .0
    else:
        for i, mask_rate in enumerate(mask_rates): # mask sequence / temporal masking
            total_masked = 0
            need_masked = int(round(mask_rate * T))
            while total_masked < need_masked:
                center = torch.randint(0, T, (1,)).item()
                if total_masked < need_masked - max_mask_len:
                    length = torch.randint(1, max_mask_len+1, (1,)).item()
                else:
                    length = need_masked - total_masked
                    
                left = max(0, center - length // 2)
                right = min(T, left + length)

                mask[:, left:right, i] *= .0
                total_masked = int(T - torch.sum(mask[0, :, i]).item())
    mask = mask.unsqueeze(-1)
    mask = mask.repeat(1, 1, 1, data_dim)
    return x_using * mask


def root_trajectory_masking(x, mask_rate=0.25, max_mask_len=15):
    """
    Applies masking to a single root trajectory for data augmentation.
    The input is expected to be a 3D tensor of shape [B, T, D].

    The masking strategy is probabilistic:
    - 25% chance: No masking is applied.
    - 75% chance: Sequential masking is applied, where segments of the
                  trajectory are masked based on the mask_rate.

    Args:
        x (Tensor): The input trajectory tensor of shape [B, T, D].
        mask_rate (float): The proportion of the sequence to mask during
                           sequential masking.
        max_mask_len (int): The maximum length of a single masked segment.

    Returns:
        Tensor: The masked trajectory tensor.
    """
    x_using = x.clone()
    B, T, D = x_using.shape

    rand_number = random.random()

    if rand_number < 0.25:
        # 25% chance: No masking
        return x_using
    else:
        # 75% chance: Sequential masking / temporal masking
        
        # mask_rate = random.randrange(20, 70)/100 # 20-70% trajectory masking
        
        mask = torch.ones(B, T, device=x.device)
        
        need_masked = int(round(mask_rate * T))
        total_masked = 0

        while total_masked < need_masked:
            center = torch.randint(0, T, (1,)).item()
            if total_masked < need_masked - max_mask_len:
                length = torch.randint(1, max_mask_len+1, (1,)).item()
            else:
                length = need_masked - total_masked
            
            left = max(0, center - length // 2)
            right = min(T, left + length)
            
            mask[:, left:right] = 0.0
            
            total_masked = T - int(mask[0, :].sum().item())

        mask = mask.unsqueeze(-1).repeat(1, 1, D)
        return x_using * mask