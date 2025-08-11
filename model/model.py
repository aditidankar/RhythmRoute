from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like, root_trajectory_masking


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# Trajectory Encoder
class TrajectoryTransformerEncoder(nn.Module):
    def __init__(
        self, 
        seq_len=150, 
        input_dim=3, 
        model_dim=512, 
        num_heads=4, 
        num_layers=4, 
        dropout=0.1,
        ):
        """        
        Args:
            seq_len    (int) : The length of the input sequence (150 for EDGE).
            input_dim  (int) : The dimension of the input coordinates (3 for x,y,z).
            model_dim  (int) : The main working dimension of the Transformer.
            num_heads  (int) : The number of attention heads in the Transformer.
            num_layers (int) : The number of layers in the Transformer.
            dropout  (float) : The dropout rate for the Transformer.
        """
        super(TrajectoryTransformerEncoder, self).__init__()

        self.linear_in = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(
            d_model=model_dim, 
            dropout=dropout, 
            batch_first=True
            )
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
            
    def forward(self, x):
        # x: [B, 150, 3]
        # Project the input trajectory to the model dimension
        x = self.linear_in(x)  # Shape: [B, 150, 512]
        
        x = self.pos_encoder(x) # Shape: [B, 150, 512]
        trajectory_features = self.transformer_encoder(x) # Shape: [B, 150, 512]
        return trajectory_features


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,    # 5 seconds, 30 fps
        latent_dim: int = 256, # 512
        ff_size: int = 1024,
        num_layers: int = 4, # 8
        num_heads: int = 4,  # 8
        dropout: float = 0.1,
        music_feature_dim: int = 4800,
        trajectory_feature_dim: int = 3,
        mask_rate: float = 0.25,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats
        self.mask_rate = mask_rate
        self.gt_trajectory_tokens = None
        self.trajectory_output = None

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        
        # music projection
        self.music_projection = nn.Linear(music_feature_dim, latent_dim)
        
        # music encoder
        self.music_encoder = nn.Sequential()
        for _ in range(2):
            self.music_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        
        # trajectory encoder
        self.trajectory_encoder = TrajectoryTransformerEncoder(
            seq_len=seq_len,
            input_dim=trajectory_feature_dim,
            model_dim=latent_dim,
            num_heads=4,
            num_layers=4,
            dropout=dropout,
        )

        # conditional projection
        self.cond_projection = nn.Linear(seq_len * 2, seq_len)
            
        # cross modal encoder
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
                        
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)

    def guided_forward(self, x, cond_embed, times, guidance_weight):
        unc, _, _ = self.forward(x, cond_embed, times, cond_drop_prob=1)
        conditioned, _, _ = self.forward(x, cond_embed, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def forward(
        self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device

        # project to latent space
        x = self.input_projection(x)
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed  = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")
                
        cond_embed_music      = cond_embed["music"]      # Shape: [B, 150, 4800]
        cond_embed_trajectory = cond_embed["trajectory"] # Shape: [B, 150, 3]
        
        # trajectory encoding
        masked_trajectory     = root_trajectory_masking(cond_embed_trajectory, mask_rate=self.mask_rate) # Shape: [B, 150, 3]
        trajectory_tokens     = self.trajectory_encoder(masked_trajectory)                               # Shape: [B, 150, 512]

        # music encoding
        music_tokens = self.music_projection(cond_embed_music) # Shape: [B, 150, 512]
        music_tokens = self.abs_pos_encoding(music_tokens)     # Shape: [B, 150, 512]
        music_tokens = self.music_encoder(music_tokens)        # Shape: [B, 150, 512]

        # fuse music and trajectory tokens
        cond_tokens = torch.cat((music_tokens, trajectory_tokens), dim=1) # Shape: [B, 300, 512]
        cond_tokens = cond_tokens.permute(0, 2, 1)                        # Shape: [B, 512, 300]
        cond_tokens = self.cond_projection(cond_tokens)                   # Shape: [B, 512, 150]
        cond_tokens = cond_tokens.permute(0, 2, 1)                        # Shape: [B, 150, 512]
        cond_tokens = self.abs_pos_encoding(cond_tokens)                  # Shape: [B, 150, 512]
        cond_tokens = self.cond_encoder(cond_tokens)                      # Shape: [B, 150, 512]
        
        # guidance dropout
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens     = torch.where(keep_mask_embed, cond_tokens, null_cond_embed) # Shape: [B, 150, 512]

        # Global conditioning for FiLM
        mean_pooled_cond_tokens = cond_tokens.mean(dim=1)                    # Shape: [B, 512]
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens) # Shape: [B, 512]
        
        # create the diffusion timestep embedding, add the extra conditional projection
        t_hidden = self.time_mlp(times) # Shape: [B, 2048]

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)          # Shape: [B, 512]
        t_tokens = self.to_time_tokens(t_hidden) # Shape: [B, 2, 512]

        # FiLM conditioning: add global context to diffusion timestep
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden) # Shape: [B, 512]
        t += cond_hidden                                                           # Shape: [B, 512]

        # Prepare for cross-attention: concatenate all context sources
        c = torch.cat((cond_tokens, t_tokens), dim=1) # Shape: [B, 152, 512]
        cond_tokens = self.norm_cond(c)               # Shape: [B, 152, 512]

        # Pass through the transformer decoder
        output = self.seqTransDecoder(x, cond_tokens, t) # Shape: [B, 150, 512]

        output = self.final_layer(output) # Shape: [B, 150, 151]
        
        # store the ground truth trajectory tokens for loss calculation
        gt_trajectory_tokens = self.trajectory_encoder(cond_embed_trajectory).detach()
        
        # Extract the trajectory output
        trajectory_output = output[:, :, 4:7]                                # Shape: [B, 150, 3]
        trajectory_output = self.trajectory_encoder(trajectory_output)  # Shape: [B, 150, 512]
        
        return output, trajectory_output, gt_trajectory_tokens
