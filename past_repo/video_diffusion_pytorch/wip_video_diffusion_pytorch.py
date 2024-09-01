# Source: video-diffusion-pytorch/video_diffusion_pytorch/video_diffusion_pytorch.py
# This code was copied from lucidrains's video-diffusion-pytorch project, specifically the video-diffusion-pytorch.py file.
# For more details and license information, please refer to the original repository:
# Repository URL: https://github.com/lucidrains/video-diffusion-pytorch

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import UCF101
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from video_diffusion_pytorch.text import tokenize, bert_embed, BERT_MODEL_DIM
from video_diffusion_pytorch.s4d import S4D ### Changed ###
# from video_diffusion_pytorch.s4 import S4Block ### Changed ###

from video_diffusion_pytorch.long_video_datasets import MineRLDataset, GQNMazesDataset, CarlaDataset ### Changed ###

# from s5 import S5, S5Block ### Changed ###

import wandb ### Changed ###

# helpers functions

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

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

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)
    
### Changed ###
class TemporalLinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)
        
    def forward(
        self, 
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
        
        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # Linear attention computes attention differently. 
        # Instead of a softmax over the dot product of queries and keys,
        # it uses a feature map to reduce dimensionality and compute attention in linear time.
        q = q.softmax(dim=-1) # Assuming the feature map is a simple softmax.
        k = k.softmax(dim=-2) # Similarly, assuming the feature map is a softmax for the keys.
        
        # scale
        
        q = q * self.scale

        # Contextual information as weighted sum of values.
        context = einsum('... h n d, ... h n e -> ... h d e', k, v)
        
        # Linear attention here with feature maps applied to queries and context.
        out = einsum('... h n d, ... h d e -> ... h n e', q, context)
        
        # Combine heads
        out = rearrange(out, '... h n d -> ... n (h d)')

        return self.to_out(out)
        

### Changed ###
class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        return x
    
### Changed ###
class LSTM(nn.Module):
    def __init__(
        self, 
        dim, 
        hidden_dim, 
        bidirectional = False, 
        num_layers = 1, 
        dropout = 0.
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            dim, 
            hidden_dim, 
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), dim)
        # self.fc1 = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(
        self, 
        x, 
        pos_bias = None, 
        focus_present_mask = None
    ):
        device = x.device
        
        c0 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), x.shape[0], self.lstm.hidden_size, device=device)
        h0 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), x.shape[0], self.lstm.hidden_size, device=device)
        
        # split out heads
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(F.gelu(x))
        # x = self.fc1(F.gelu(x))
        # x = self.fc2(F.gelu(x))
        
        return x

# ### Changed ###
# class S4Layer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         hidden_dim,
#         bidirectional = False,
#         rotary_emb = None
#     ):
#         super().__init__()
#         self.bidirectional = bidirectional

#         self.s4 = S4(dim, hidden_dim, transposed=False)
#         if bidirectional:
#             self.s4_rev = S4(dim, hidden_dim, transposed=False)
#         self.fc = nn.Linear(dim * (2 if bidirectional else 1), dim)
    
#     def forward(
#         self,
#         x,
#         pos_bias = None,
#         focus_present_mask = None
#     ):
#         device = x.device
        
#         # split out heads
#         x_, _ = self.s4(x)
#         if self.bidirectional:
#             x_rev, _ = self.s4_rev(torch.flip(x, [1]))
#             x_ = torch.cat((x_, torch.flip(x_rev, [1])), dim=-1)
#         x = self.fc(x_)

#         return x

### Changed ###
class S4DLayer(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        linear_dim,
        version,
        bidirectional = False,
        rotary_emb = None
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.version = version

        # S4D settings 
        if version in range(1, 16) or version == 20:
            self.s4d = S4D(dim, hidden_dim, transposed=False)
            if bidirectional:
                self.s4d_rev = S4D(dim, hidden_dim, transposed=False)
        elif version in range(16, 19) or version == 21:
            self.s4d = S4D(dim, hidden_dim, transposed=False, output_glu=False)
            if bidirectional:
                self.s4d_rev = S4D(dim, hidden_dim, transposed=False, output_glu=False)
        elif version == 19 or version == 22:
            expand = 2
            self.s4d = S4D(expand * dim, hidden_dim, transposed=False, output_glu=False)
            if bidirectional:
                self.s4d_rev = S4D(expand * dim, hidden_dim, transposed=False, output_glu=False)
            
        # Linear settings 
        if version == 6 or version == 7 or version == 13 or version == 15:
            self.fc0 = nn.Linear(dim, dim)                                     # ver6 (addfc12.4) # ver7 (addfc12.5)
        elif version == 16:                                                    # ver16
            self.fc01 = nn.Linear(dim, linear_dim)                              
            self.fc02 = nn.Linear(linear_dim, dim)                              
        
        if version == 2 or version == 4:
            self.fc1 = nn.Linear(dim * (2 if bidirectional else 1), linear_dim) # ver2 (addfc12)
            self.fc2 = nn.Linear(linear_dim, dim)                               # ver2 (addfc12)
        elif version == 3 or version == 8 or version == 11 or version == 12 or version == 16:
            self.fc1 = nn.Linear(dim, linear_dim)                               # ver3 (addfc12.1) # ver4 (addfc12.2)
            self.fc2 = nn.Linear(linear_dim, dim)                               # ver3 (addfc12.1) # ver4 (addfc12.2)
        elif version == 5 or version == 7:
            self.fc1 = nn.Linear(dim * (2 if bidirectional else 1), dim)        # ver5 (addfc12.3) # ver7 (addfc12.5)
        elif version == 14 or version == 15:
            self.fc1 = nn.Linear(dim, dim)                                      # ver14 (addfc12.8) # ver15 (addfc12.9)
        elif version == 20:
            self.fc1 = nn.Linear(dim, dim * 2)                    
            self.fc2 = nn.Linear(dim * 2, dim)                    
            
        if version == 9 or version == 17:   # GSS or BiGS
            self.fc_shortcut = nn.Linear(dim, 3 * dim)    
                                      
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            
            if bidirectional:
                self.fc_rev1 = nn.Linear(dim, dim)
                self.fc_rev2 = nn.Linear(dim, dim)
            
            self.fc3 = nn.Linear(dim, 3 * dim)
            
            self.fc_out = nn.Linear(3 * dim, dim)
            
        if version == 10:
            self.fc_shortcut = nn.Linear(dim, 2 * dim)    
                                      
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(2 * dim, 2 * dim)
            
            self.fc_out = nn.Linear(2 * dim, dim)
            
        # if version == 18:
        #     self.fc_qkv = nn.Linear(dim, 3 * dim) 
            
        #     d_conv = 4
        #     self.conv = nn.Conv1d(in_channels=dim,
        #                     out_channels=dim,
        #                     kernel_size=d_conv,
        #                     groups=dim,
        #                     padding=d_conv - 1)
            
        #     self.fc_out = nn.Linear(dim, dim)
            
        if version == 19: # Mamba or Vim
            expand = 2
            d_conv = 4
            
            self.fc_init = nn.Linear(dim, 2 * expand * dim) 
            
            self.conv = nn.Conv1d(in_channels=dim * expand,
                            out_channels=dim * expand,
                            kernel_size=d_conv,
                            groups=dim * expand,
                            padding=d_conv - 1)
            
            if bidirectional:
                self.conv_rev = nn.Conv1d(in_channels=dim * expand,
                                    out_channels=dim * expand,
                                    kernel_size=d_conv,
                                    groups=dim * expand,
                                    padding=d_conv - 1)
            
            self.fc_out = nn.Linear(dim * expand, dim)
            
        if version == 21: # GSS, BiGS with bigger MLP                                                       
            self.fc_shortcut = nn.Linear(dim, linear_dim)    
                                      
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            
            if bidirectional:
                self.fc_rev1 = nn.Linear(dim, dim)
                self.fc_rev2 = nn.Linear(dim, dim)
            
            self.fc3 = nn.Linear(dim, linear_dim)
            
            self.fc_out = nn.Linear(linear_dim, dim)
            
        if version == 22: # Mamba, Vim with bigger MLP
            expand = 2
            d_conv = 4
            
            self.fc_init = nn.Linear(dim, 2 * expand * dim) 
            
            self.conv = nn.Conv1d(in_channels=dim * expand,
                            out_channels=dim * expand,
                            kernel_size=d_conv,
                            groups=dim * expand,
                            padding=d_conv - 1)
            
            if bidirectional:
                self.conv_rev = nn.Conv1d(in_channels=dim * expand,
                                    out_channels=dim * expand,
                                    kernel_size=d_conv,
                                    groups=dim * expand,
                                    padding=d_conv - 1)
            
            self.fc_out_1 = nn.Linear(dim * expand, linear_dim)
            self.fc_out_2 = nn.Linear(linear_dim, dim)
    
    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        device = x.device
        
        # split out heads
        if self.version in range(1, 9) or self.version in range(11, 17) or self.version == 20:
            if self.version == 6 or self.version == 7:
                x = self.fc0(x)                          # ver6 (addfc12.4) # ver7 (addfc12.5)
            elif self.version == 12:                       # ver12
                x = self.fc1(x)                   
                x = self.fc2(F.gelu(x))        
            elif self.version == 13 or self.version == 15: # ver13 # ver15       
                x = F.gelu(self.fc0(x))
            elif self.version == 16:                       # ver16
                x = self.fc01(x)
                x = self.fc02(F.gelu(x))
            elif self.version == 1 or self.version == 2 or self.version == 3 or self.version == 4 \
            or self.version == 5 or self.version == 8 or self.version == 11 or self.version == 14 \
            or self.version == 20:
                pass
            else:
                raise NotImplementedError()
                                 
            x_, _ = self.s4d(x)
            if self.bidirectional:
                x_rev, _ = self.s4d_rev(torch.flip(x, [1]))
                if self.version == 1 or self.version == 3 or self.version == 6 or self.version == 8 \
                or self.version == 11 or self.version == 12 or self.version == 13 or self.version ==14 \
                or self.version == 15 or self.version == 16 or self.version == 20:
                    x_ = x_ + torch.flip(x_rev, [1])      # ver1 (rmfc)    # ver3 (rmfc12.1)
                elif self.version == 2 or self.version == 4 or self.version == 5 or self.version == 7:
                    x_ = torch.cat((x_, torch.flip(x_rev, [1])), dim=-1)   # ver2 (addfc12)
                else:
                    raise NotImplementedError()
            
            if self.version == 1 or self.version == 6 or self.version == 12 or self.version == 13:
                x = x_                                    # ver1 (rmfc) 
            elif self.version == 2 or self.version == 3:
                x = self.fc1(F.gelu(x_))                  # ver2 (addfc12) # ver3 (addfc12.1)
                x = self.fc2(F.gelu(x))                   # ver2 (addfc12) # ver3 (addfc12.1)
            elif self.version == 4 or self.version == 8 or self.version == 16 or self.version == 20:
                x = self.fc1(x_)                          # ver4 (addfc12.2) # ver8 (addfc12.6)
                x = self.fc2(F.gelu(x))                   # ver4 (addfc12.2) # ver8 (addfc12.6)
            elif self.version == 5 or self.version == 7:
                x = self.fc1(x_)                          # ver5 (addfc12.3)
            elif self.version == 11:
                x = self.fc1(x_)                          # ver11 (addfc12.7)
                x = self.fc2(F.gelu(x))                   # ver11 (addfc12.7)
                x = x + x_                                # ver11 (addfc12.7)
            elif self.version == 14 or self.version == 15:
                x = F.gelu(self.fc1(x_))                  # ver14 (addfc12.8) # ver15 (addfc12.9)
            else:
                raise NotImplementedError()
                
        elif self.version == 9: 
            # shortcut
            x_shortcut = self.fc_shortcut(x)
            x_shortcut = F.gelu(x_shortcut)
            
            # forward ssm
            x_ = F.gelu(self.fc1(x))
            x_, _ = self.s4d(x_)
            x_ = self.fc2(x_)
            
            if self.bidirectional:
                # backward ssm
                x_rev = F.gelu(self.fc_rev1(torch.flip(x, [1])))
                x_rev, _ = self.s4d_rev(x_rev)
                x_rev = torch.flip(self.fc_rev2(x_rev), [1])
            
                # element-wise multiplication with ssms
                x_ = x_ * x_rev
                
            x_ = self.fc3(x_)
            x_ = F.gelu(x_)
            
            # element-wise multiplication with shortcut
            x = x_shortcut * x_
            x = self.fc_out(x)
        
        elif self.version == 10:
            if not self.bidirectional:
                raise ValueError('S4DLayer version 10 must be bidirectional')
            
            # shortcut
            x_shortcut = self.fc_shortcut(x)
            x_shortcut = F.gelu(x_shortcut)
            
            # ssms
            x = F.gelu(self.fc1(x))
            
            # forward ssm
            x_, _ = self.s4d(x)
            # backward ssm
            x_rev, _ = self.s4d_rev(torch.flip(x, [1]))
            
            # ssms concat
            x_ = torch.cat((x_, torch.flip(x_rev, [1])), dim=-1)
            x_ = F.gelu(self.fc2(x_))
            
            # element-wise multiplication with shortcut
            x = x_shortcut * x_
            x = self.fc_out(x)
            
        elif self.version == 17 or self.version == 21: ### GSS, BiGS ###
            # shortcut
            x_shortcut = self.fc_shortcut(x)
            x_shortcut = F.gelu(x_shortcut)
            
            # forward ssm
            x_ = F.gelu(self.fc1(x))
            x_, _ = self.s4d(x_)
            x_ = self.fc2(x_)
            
            if self.bidirectional:
                # backward ssm
                x_rev = F.gelu(self.fc_rev1(torch.flip(x, [1])))
                x_rev, _ = self.s4d_rev(x_rev)
                x_rev = torch.flip(self.fc_rev2(x_rev), [1])
            
                # element-wise multiplication with ssms
                x_ = x_ * x_rev
                
            x_ = self.fc3(x_)
            x_ = F.gelu(x_)
            
            # element-wise multiplication with shortcut
            x = x_shortcut * x_
            x = self.fc_out(x)
            
        # elif self.version == 18: ### H3 ###
        #     if self.bidirectional:
        #         raise ValueError('S4DLayer version 18 must be unidirectional')
        #     num_frames = x.shape[1]
            
        #     x_q, x_k, x_v = self.fc_qkv(x).chunk(3, dim=-1)

        #     x_k = x_k.transpose(2,1)
        #     x_k = self.conv(x_k)[..., :num_frames]
        #     x_k = x_k.transpose(2,1)
            
        #     x_prod = x_k * x_v
        #     x_ssm, _ = self.s4d(x_prod)
            
        #     x_out = x_q * x_ssm
            
        #     x = self.fc_out(x_out)
            
        elif self.version == 19: ### Mamba, Bidirectional Mamba ###
            num_frames = x.shape[1]
            
            x_, x_shortcut = self.fc_init(x).chunk(2, dim=-1)
            
            x_in = x_.transpose(2, 1)
            x_shifted = F.silu(self.conv(x_in)[..., :num_frames])
            x_shifted = x_shifted.transpose(2, 1)
            
            x_ssm, _ = self.s4d(x_shifted)
            
            x_out = x_ssm * F.silu(x_shortcut)
            
            if self.bidirectional:
                x_in_rev = torch.flip(x_, [1])
                
                x_in_rev = x_in_rev.transpose(2, 1)
                x_shifted_rev = F.silu(self.conv_rev(x_in_rev)[..., :num_frames])
                x_shifted_rev = x_shifted_rev.transpose(2, 1)
                
                x_ssm_rev, _ = self.s4d_rev(x_shifted_rev)
                
                x_ssm_rev = torch.flip(x_ssm_rev, [1])
                
                x_out_rev = x_ssm_rev * F.silu(x_shortcut)
                
                # element-wise summation with ssms
                x_out = x_out + x_out_rev
            
            x = self.fc_out(x_out)
            
        elif self.version == 22: ### Mamba, Bidirectional Mamba ###
            num_frames = x.shape[1]
            
            x_, x_shortcut = self.fc_init(x).chunk(2, dim=-1)
            
            x_in = x_.transpose(2, 1)
            x_shifted = F.silu(self.conv(x_in)[..., :num_frames])
            x_shifted = x_shifted.transpose(2, 1)
            
            x_ssm, _ = self.s4d(x_shifted)
            
            x_out = x_ssm * F.silu(x_shortcut)
            
            if self.bidirectional:
                x_in_rev = torch.flip(x_, [1])
                
                x_in_rev = x_in_rev.transpose(2, 1)
                x_shifted_rev = F.silu(self.conv_rev(x_in_rev)[..., :num_frames])
                x_shifted_rev = x_shifted_rev.transpose(2, 1)
                
                x_ssm_rev, _ = self.s4d_rev(x_shifted_rev)
                
                x_ssm_rev = torch.flip(x_ssm_rev, [1])
                
                x_out_rev = x_ssm_rev * F.silu(x_shortcut)
                
                # element-wise summation with ssms
                x_out = x_out + x_out_rev
            
            x = F.silu(self.fc_out_1(x_out))
            x = self.fc_out_2(x)

        return x
    
### Changed ###
# class S5Layer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         hidden_dim,
#         bidirectional = False,
#         rotary_emb = None
#     ):
#         super().__init__()
#         self.bidirectional = bidirectional

#         self.s5 = S5(dim, hidden_dim)
#         if bidirectional:
#             self.s5_rev = S5(dim, hidden_dim)
#         self.fc = nn.Linear(dim * (2 if bidirectional else 1), dim)
    
#     def forward(
#         self,
#         x,
#         pos_bias = None,
#         focus_present_mask = None
#     ):
#         device = x.device
        
#         # split out heads
#         x_ = self.s5(x)
#         if self.bidirectional:
#             x_rev = self.s5_rev(torch.flip(x, [1]))
#             x_ = torch.cat((x_, torch.flip(x_rev, [1])), dim=-1)
#         x = self.fc(x_)
        
#         return x

# model

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        timeemb_linears = 2,
        attn_heads = 8, 
        attn_dim_head = 64, ### Changed ### 32 -> 64
        s4d_hidden_dim = None, ### Changed ### 
        s4d_linear_dim = None, ### Changed ###
        use_bert_text_cond = False,
        init_dim = None, 
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        block_type = 'resnet',
        resnet_groups = 8,
        temporal_arch = 'attn',
        s4d_version = None, ### Changed ###
    ):
        super().__init__()
        self.channels = channels

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        ### Changed ###
        if temporal_arch == 'attn':
            temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb)) 
        elif temporal_arch == 'l-attn':
            temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c', TemporalLinearAttention(dim, heads = attn_heads, dim_head = attn_dim_head, rotary_emb = rotary_emb))
        elif temporal_arch == 'nan':
            temporal_layer = lambda dim: IdentityLayer()
        elif temporal_arch == 'lstm' or temporal_arch == 'bi-lstm':
            if temporal_arch == 'lstm':
                temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', '(b h w) f c', LSTM(dim, attn_dim_head*attn_heads))
            else:
                temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', '(b h w) f c', LSTM(dim, attn_dim_head*attn_heads, bidirectional=True))
        # elif temporal_arch == 's4' or temporal_arch == 'bi-s4':
        #     if temporal_arch == 's4': 
        #         temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', '(b h w) f c', S4Layer(dim, attn_dim_head*attn_heads)) 
        #     else:
        #         temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', '(b h w) f c', S4Layer(dim, attn_dim_head*attn_heads, bidirectional=True))
        elif temporal_arch == 's4d' or temporal_arch == 'bi-s4d':
            if s4d_hidden_dim is None:
                s4d_hidden_dim = attn_dim_head*attn_heads
            if s4d_linear_dim is None:
                s4d_linear_dim = attn_dim_head*attn_heads
                
            if s4d_version == None:
                raise ValueError('s4d_version not specified')

            if temporal_arch == 's4d': 
                temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', '(b h w) f c', S4DLayer(dim, s4d_hidden_dim, s4d_linear_dim, version=s4d_version))
            else:
                temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', '(b h w) f c', S4DLayer(dim, s4d_hidden_dim, s4d_linear_dim, bidirectional=True, version=s4d_version))
        # elif temporal_arch == 's5' or temporal_arch == 'bi-s5':
            # if temporal_arch == 's5': 
            #     temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', '(b h w) f c', S5Layer(dim, attn_dim_head*attn_heads)) 
            # else:
            #     temporal_layer = lambda dim: EinopsToAndFrom('b c f h w', '(b h w) f c', S5Layer(dim, attn_dim_head*attn_heads, bidirectional=True))
        else:
            raise ValueError('temporal_layer not implemented')
            
        self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_layer = Residual(PreNorm(init_dim, temporal_layer(init_dim))) ### Changed ###

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        ### Changed ###
        time_dim = dim * 32 # dim * 4 -> dim * 32
        # self.time_mlp = nn.Sequential(
        #     SinusoidalPosEmb(dim),
        #     nn.Linear(dim, time_dim),
        #     nn.GELU(),
        #     nn.Linear(time_dim, time_dim)
        # )
        layers = [SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU()]
        for _ in range(timeemb_linears - 2):
            layers += [nn.Linear(time_dim, time_dim), nn.GELU()]
        layers.append(nn.Linear(time_dim, time_dim))
        self.time_mlp = nn.Sequential(*layers)

        # text conditioning

        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim

        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_layer(dim_out))), ### Changed ###
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads = attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_layer = Residual(PreNorm(mid_dim, temporal_layer(mid_dim))) ### Changed ###

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads = attn_heads))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_layer(dim_in))), ### Changed ###
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond = None,
        null_cond_prob = 0.,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)

        x = self.init_conv(x)

        x = self.init_temporal_layer(x, pos_bias = time_rel_pos_bias) ### Changed ###

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device = device)
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            t = torch.cat((t, cond), dim = -1)

        h = []

        for block1, block2, spatial_attn, temporal_layer, downsample in self.downs: ### Changed ###
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_layer(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask) ### Changed ###
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_layer(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask) ### Changed ###
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_layer, upsample in self.ups: ### Changed ###
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_layer(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask) ### Changed ###
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls = False,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        # x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale))
        x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.module.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale)) ### Changed ### # for DP (old version)

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond = None, cond_scale = 1., batch_size = 16):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond = cond, cond_scale = cond_scale)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr = self.text_use_bert_cls)
            cond = cond.to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c = self.channels, f = self.num_frames, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)

# trainer class

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 16,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform = self.transform)
        return self.cast_num_frames_fn(tensor)

### Changed ###
def custom_video_collate_fn(batch):
    # バッチ内のビデオデータを取得
    videos = [item[0] for item in batch]  # item[0] はビデオデータ

    # ビデオデータのテンソルリストを作成
    videos = [torch.tensor(v) for v in videos]

    # ビデオデータをパディング
    videos_padded = pad_sequence(videos, batch_first=True, padding_value=0)

    return videos_padded

### Changed ###
import numpy as np
class MovingMNISTDataset(Dataset):
    def __init__(self, file_path, image_size=64):
        self.data = np.load(file_path)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),  # リサイズ
            T.ToTensor(),  # テンソルに変換
            T.Normalize((0.5,), (0.5,))  # 正規化: 平均0.5, 標準偏差0.5でデータを[-1, 1]の範囲にスケール
        ])

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        frames = self.data[:, idx]
        frames = [Image.fromarray(frame) for frame in frames]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack the images into a single tensor
        frames = torch.stack(frames, dim=0)
        return frames.permute(1,0,2,3)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        folder,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        beta1 = 0.9,
        beta2 = 0.999,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        num_sample_rows = 2, ### Changed ###
        max_grad_norm = None,
        num_gpus = 1 ### Changed ###
    ):
        super().__init__()
        self.model = diffusion_model
        # self.model = diffusion_model.module ### Changed ### # for DP (new version)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        # self.image_size = diffusion_model.module.image_size ### Changed ### # for DP (new version)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        
        image_size = diffusion_model.image_size 
        channels = diffusion_model.channels 
        num_frames = diffusion_model.num_frames 
        # image_size = diffusion_model.module.image_size ### Changed ### # for DP (new version)
        # channels = diffusion_model.module.channels ### Changed ### # for DP (new version)
        # num_frames = diffusion_model.module.num_frames ### Changed ### # for DP (new version)

        ### Changed ###
        self.dataset = dataset
        # self.ds = Dataset(folder, image_size, channels = channels, num_frames = num_frames)
        if dataset == 'movingmnist':
            self.ds = MovingMNISTDataset(folder, image_size=image_size)
        elif dataset == 'ucf101-all':
            # if num_frames != 16:
            #     raise ValueError('UCF101 dataset should be 16 frames')
            ds_train = UCF101(
                root=folder + '/UCF-101',
                annotation_path=folder + '/ucfTrainTestlist',
                frames_per_clip=num_frames,
                train=True,
                transform=T.Compose([
            T.Lambda(lambda x: x / 255.),
            T.Lambda(lambda x: x.permute(3, 0, 1, 2)),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            ])
            )
            ds_test = UCF101(
                root=folder + '/UCF-101',
                annotation_path=folder + '/ucfTrainTestlist',
                frames_per_clip=num_frames,
                train=False,
                transform=T.Compose([
            T.Lambda(lambda x: x / 255.),
            T.Lambda(lambda x: x.permute(3, 0, 1, 2)),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            ])
            )
            self.ds = data.ConcatDataset([ds_train, ds_test])
        elif dataset == 'minerl':
            self.ds = MineRLDataset(
                path=folder, 
                shard=0, 
                num_shards=1, 
                T=num_frames,
                image_size = image_size
            )
        elif dataset == 'gqn-mazes':
            self.ds = GQNMazesDataset(
                path=folder, 
                shard=0, 
                num_shards=1, 
                T=num_frames,
                image_size=image_size)
        else:
            raise ValueError('dataset not implemented')

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        ### Changed ###
        if dataset == 'movingmnist':
            self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, drop_last=True))
        elif dataset == 'ucf101-all':
            self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, drop_last=True, collate_fn=custom_video_collate_fn))
        elif dataset == 'minerl':
            self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, drop_last=True))
        elif dataset == 'gqn-mazes':
            self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True, drop_last=True))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas=(beta1, beta2)) ### Changed ###
        # self.opt = Adam(diffusion_model.module.parameters(), lr = train_lr) ### Changed ### # for DP (new version)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()
        
        self.num_gpus = num_gpus ### Changed ###

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt' : self.opt.state_dict(), 
            'scaler': self.scaler.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], **kwargs)
        self.ema_model.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])
        try:
            self.opt.load_state_dict(data['opt'], **kwargs)
        except:
            print('Warning! optimizer is not loaded')

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop
    ):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                if self.dataset == 'ucf101-all' or self.dataset == 'movingmnist':
                    data = next(self.dl).cuda()
                else:
                    data = next(self.dl)[0].permute(0, 2, 1, 3, 4).cuda()

                with autocast(enabled = self.amp):
                    loss = self.model(
                        data,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f'{self.step}: {loss.item()}')

            log = {self.dataset + '/loss': loss.item()} 
            
            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size//self.num_gpus) ### Changed ###

                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = torch.cat(all_videos_list, dim = 0)

                all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

                one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
                video_path = str(self.results_folder / str(f'{milestone}.gif'))
                video_tensor_to_gif(one_gif, video_path)
                log = {**log, self.dataset + '/sample': wandb.Image(video_path)} ### Changed ###
                self.save(milestone) 

            log_fn(log)
            self.step += 1

        print('training completed')