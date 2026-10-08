import math
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out=None, conv_groups=1):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1, groups=conv_groups)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

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

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, conv_groups=1):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1, groups=conv_groups)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class SpatialModulation(nn.Module):
    def __init__(self, dim_ctx, dim_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(dim_ctx, dim_out, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out * 2, 3, padding=1)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, ctx):
        scale_shift = self.mlp(ctx)
        scale, shift = scale_shift.chunk(2, dim=1)
        return scale, shift

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, conv_groups=1, use_spatial_mod=False, dim_ctx=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.use_spatial_mod = use_spatial_mod
        if use_spatial_mod:
            self.spatial_mod = SpatialModulation(default(dim_ctx, dim), dim_out)

        self.block1 = Block(dim, dim_out, groups=groups, conv_groups=conv_groups)
        self.block2 = Block(dim_out, dim_out, groups=groups, conv_groups=conv_groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1, groups=conv_groups) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, ctx=None):
        scale_shift = None
        
    
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

       
        if self.use_spatial_mod and exists(ctx):
            spatial_scale, spatial_shift = self.spatial_mod(ctx)
            if scale_shift is not None:
                scale, shift = scale_shift
                scale_shift = (scale + spatial_scale, shift + spatial_shift)
            else:
                scale_shift = (spatial_scale, spatial_shift)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)
    
class TemporalAttention(nn.Module):
    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x
    
class TemporalAttentionModule(nn.Module):
    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.hidden_dim = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, self.hidden_dim, bias=False),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        f_x = self.conv1(attn)
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u
    
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, n_layer=1):
        super().__init__()
        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.cur_states = [None for _ in range(n_layer)]
        self.n_layer = n_layer
        self.conv_gates = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=2 * self.hidden_dim,
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

        self.conv_cans = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=input_dim + hidden_dim if i == 0 else hidden_dim * 2,
                    out_channels=self.hidden_dim,
                    kernel_size=kernel_size,
                    padding=self.padding,
                )
                for i in range(n_layer)
            ]
        )

    def init_hidden(self, batch_shape, device):
        b, _, h, w = batch_shape
        for i in range(self.n_layer):
            self.cur_states[i] = torch.zeros((b, self.hidden_dim, h, w), device=device)

    def get_state(self):
        return [None if state is None else state.clone() for state in self.cur_states]

    def set_state(self, states):
        if len(states) != self.n_layer:
            raise ValueError(f'Expected {self.n_layer} GRU states, got {len(states)}')
        self.cur_states = [None if state is None else state.clone() for state in states]

    def step_forward(self, input_tensor, index):
        h_cur = self.cur_states[index]
        assert h_cur is not None
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates[index](combined)

        reset_gate, update_gate = torch.split(torch.sigmoid(combined_conv), self.hidden_dim, dim=1)
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_cans[index](combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        self.cur_states[index] = h_next
        return h_next
    
    def forward(self, input_tensor):
        for i in range(self.n_layer):
            input_tensor = self.step_forward(input_tensor, i)
        return input_tensor


class WaveletContextEncoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        
        self.init_conv = nn.Conv2d(channels, dim, 7, padding = 3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.downs = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) -1 )
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in),
                        ConvGRUCell(dim_in, dim_in, 3, n_layer=1),
                        Downsample(dim_in, dim_out) if not is_last else nn.Identity()
                    ]
                )
            )

    def init_state(self, shape, device):
        for i, ml in enumerate(self.downs):
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape, device)

    def get_state(self):
        return [ml[1].get_state() for ml in self.downs]

    def set_state(self, states):
        if len(states) != len(self.downs):
            raise ValueError(f'Expected {len(self.downs)} context states, got {len(states)}')
        for ml, state in zip(self.downs, states):
            ml[1].set_state(state)
            
    def forward(self, x):
        x = self.init_conv(x)
        context = []
        for i, (resnet, conv, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = conv(x)
            context.append(x)
            x = downsample(x)
        return context
    
    def scan_ctx(self, frames, split_idx=None):
        b, t, c, h, w = frames.shape
        state_shape = (b, c, h, w)
        self.init_state(state_shape, frames.device)
        if split_idx is None:
            split_idx = t
        if not (1 <= split_idx <= t):
            raise ValueError(f'split_idx must be in [1, {t}], got {split_idx}')
        local_ctx = None
        global_ctx = None
        
        for i in range(t):
            global_ctx = self.forward(frames[:, i])
            if i == (split_idx - 1):
                local_ctx = [h.clone() for h in global_ctx]
        return global_ctx, local_ctx
        
        
class DWUNet(nn.Module):
    def __init__(
        self,
        dim=96,                  
        T_in=5,
        frame_channels=3,        
        stem_groups=3,          
        dim_mults=(1, 2, 4, 8),
        self_condition=True,
        resnet_block_groups=8,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16
    ):
        super().__init__()

        self.T_in = T_in
        self.frame_channels = frame_channels
        self.channels = T_in * frame_channels * 2
        self.self_condition = self_condition
        self.stem_groups = stem_groups
        input_channels = self.channels

        init_dim = dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3, groups=stem_groups)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.frag_idx_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            conv_groups = 3 if ind < 2 else 1

            self.downs.append(nn.ModuleList([
                
                block_klass(dim_in, dim_in, time_emb_dim=time_dim * 2, conv_groups=conv_groups, use_spatial_mod=True, dim_ctx=dim_in),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim * 2, conv_groups=conv_groups),
                Residual(PreNorm(dim_in, TemporalAttention(dim_in))),
                Downsample(dim_in, dim_out, conv_groups=conv_groups) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1, groups=conv_groups)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim * 2)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim * 2)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim * 2),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim * 2),
                Residual(PreNorm(dim_out, TemporalAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.out_dim = T_in * frame_channels

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim * 2)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1, groups=stem_groups)

    def forward(self, x, time, cond=None, ctx=None, idx=None):
        if self.stem_groups > 1:
            cond = default(cond, lambda: torch.zeros_like(x))
            x = torch.cat((cond, x), dim=1)
            x = rearrange(x, 'b t c h w -> b (c t) h w')
        else:
            x = rearrange(x, 'b t c h w -> b (t c) h w')
            if exists(cond):
                cond = rearrange(cond, 'b t c h w -> b (t c) h w')
            cond = default(cond, lambda: torch.zeros_like(x))
            x = torch.cat((cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        f_idx = self.frag_idx_mlp(idx)
        t = torch.cat((t, f_idx), dim = 1)

        h = []

        for idx_layer, (block1, block2, attn, downsample) in enumerate(self.downs):
            
            x = block1(x, t, ctx=ctx[idx_layer])
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        if self.stem_groups > 1:
            x = rearrange(x, 'b (c t) h w -> b t c h w', c=self.frame_channels, t=self.T_in)
        else:
            x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.T_in, c=self.frame_channels)
        return x

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        context_encoder,
        *,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.context_encoder = context_encoder

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be valid'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.offset_noise_strength = offset_noise_strength
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        legacy_prefix = prefix + "ctx_net."
        for key in [k for k in state_dict if k.startswith(legacy_prefix)]:
            new_key = prefix + "context_encoder." + key[len(legacy_prefix):]
            state_dict[new_key] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @property
    def device(self):
        return self.betas.device
    
    def load_backbone(self, backbone_net):
        self.backbone_net = backbone_net

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond=None, ctx=None, idx=None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, cond=cond, ctx=ctx, idx=idx)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, cond=None, ctx=None, idx=None, clip_denoised = True):
        preds = self.model_predictions(x, t, cond=cond, ctx=ctx, idx=idx,)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, cond=None, ctx=None, idx=None,):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, cond=cond, ctx=ctx, idx=idx, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, ctx=None, idx=None, return_all_timesteps = False):
        batch, device = shape[0], cond.device if cond is not None else self.device
        frames_pred = torch.randn(shape, device = device)
        imgs = [frames_pred]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            frames_pred, _ = self.p_sample(frames_pred, t, cond=cond, ctx=ctx, idx=idx)
            imgs.append(frames_pred)

        ret = frames_pred if not return_all_timesteps else torch.stack(imgs, dim = 1)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, ctx=None, idx=None, return_all_timesteps = False, record_timesteps=None):
        batch, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        device = cond.device if cond is not None else self.device
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        frames_pred = torch.randn(shape, device = device)
        
        
        if record_timesteps is not None:
            recorded_states = {}
            record_set = set(record_timesteps)
        else:
            imgs = [frames_pred]

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=True):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            
            pred_noise, x_start, *_ = self.model_predictions(frames_pred, time_cond, cond=cond, ctx=ctx, idx=idx, clip_x_start = True, rederive_pred_noise = True)

            
            if record_timesteps is not None and time in record_set:
                recorded_states[time] = frames_pred.clone()

            if time_next < 0:
                frames_pred = x_start
                if record_timesteps is not None:
                    recorded_states[0] = frames_pred.clone()
                else:
                    imgs.append(frames_pred)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(frames_pred)

            frames_pred = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if record_timesteps is None:
                imgs.append(frames_pred)

        if record_timesteps is not None:
            
            imgs_list = [recorded_states.get(t, frames_pred) for t in sorted(record_timesteps, reverse=True)]
            ret = torch.stack(imgs_list, dim=1) if return_all_timesteps else frames_pred
        else:
            ret = frames_pred if not return_all_timesteps else torch.stack(imgs, dim = 1)
        return ret

    @torch.no_grad()
    def sample(self, frames_in, T_out, return_all_timesteps = False):
        if not hasattr(self, 'backbone_net'):
            raise RuntimeError('backbone_net is not loaded. Call load_backbone() before sampling.')
        if frames_in.ndim != 5:
            raise ValueError(f'frames_in must have shape (B, T, C, H, W), got {tuple(frames_in.shape)}')

        B, T_in, c, h, w = frames_in.shape
        device = self.device
        if T_out % T_in != 0:
            raise ValueError(f'T_out ({T_out}) must be divisible by T_in ({T_in}).')
        if self.model.frame_channels != c:
            raise ValueError(
                f'Input channel mismatch: model expects {self.model.frame_channels} channels per frame, got {c}.'
            )
        if self.context_encoder.channels != c:
            raise ValueError(
                f'WaveletContextEncoder channel mismatch: context_encoder expects {self.context_encoder.channels} channels, got {c}.'
            )

        backbone_output, _ = self.backbone_net.predict(frames_in)
        expected_backbone_shape = (B, T_out, c, h, w)
        if backbone_output.shape != expected_backbone_shape:
            raise ValueError(
                f'backbone_output must have shape {expected_backbone_shape}, got {tuple(backbone_output.shape)}'
            )
        if not torch.isfinite(frames_in).all():
            raise ValueError('frames_in contains NaN or Inf values.')
        if not torch.isfinite(backbone_output).all():
            raise ValueError('backbone_output contains NaN or Inf values.')
        
        frames_in = self.normalize(frames_in)
        backbone_output = self.normalize(backbone_output)
        
        global_ctx, local_ctx = self.context_encoder.scan_ctx(torch.cat((frames_in, backbone_output), dim=1), split_idx=T_in)
        
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        
        frames_pred = []
        ys = []
        
        pre_frag = frames_in
        pre_mu = None
        for frag_idx in tqdm(range(T_out // T_in), desc="sampling frags:", disable=(frames_in.device.type != 'cuda')):
            mu = backbone_output[:, frag_idx * T_in : (frag_idx + 1) * T_in]
            cond = pre_frag - pre_mu if  pre_mu is not None else torch.zeros_like(pre_frag)

            y = sample_fn(
                (B, T_in, c, h, w), 
                cond=cond, 
                ctx=global_ctx,
                idx=torch.full((B,), frag_idx, device = device, dtype = torch.long), 
                return_all_timesteps = return_all_timesteps
                )

            frag_pred = y + mu
            frames_pred.append(frag_pred)
            ys.append(y)
            
            pre_frag = frag_pred
            pre_mu = mu
        
        frames_pred = self.unnormalize(torch.cat(frames_pred, dim=1))
        frames_pred = frames_pred.clamp(0,1)
        ys = torch.cat(ys, dim=1)
        
        backbone_output = self.unnormalize(backbone_output)
        return frames_pred, backbone_output, ys

    def predict(self, frames_in,  compute_loss=False, **kwargs):
        T_out = default(kwargs.get('T_out'), 20)
        pred, mu, y = self.sample(frames_in=frames_in, T_out=T_out)
        if compute_loss:
            raise NotImplementedError(
                "Training is implemented by the wavelet residual diffusion subclass."
            )
        return pred, None


# ---------------------------------------------------------------------------
#  Haar Wavelet Transform
# ---------------------------------------------------------------------------

def _haar_scale(haar_mode):
    if haar_mode == "legacy":
        return 4.0, 1.0
    if haar_mode == "ortho":
        return 2.0, 2.0
    raise ValueError(f"haar_mode must be 'legacy' or 'ortho', got {haar_mode}")

def dwt_2d(x, haar_mode="ortho"):
    coeff_divisor, _ = _haar_scale(haar_mode)
    x_ll = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] +
             x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / coeff_divisor
    x_lh = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] -
             x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / coeff_divisor
    x_hl = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] +
             x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / coeff_divisor
    x_hh = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] -
             x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / coeff_divisor
    return x_ll, x_lh, x_hl, x_hh

def idwt_2d(x_ll, x_lh, x_hl, x_hh, haar_mode="ortho"):
    h2, w2 = x_ll.shape[-2], x_ll.shape[-1]
    _, recon_divisor = _haar_scale(haar_mode)
    x = x_ll.new_zeros(*x_ll.shape[:-2], h2 * 2, w2 * 2)
    x[..., 0::2, 0::2] = (x_ll + x_lh + x_hl + x_hh) / recon_divisor
    x[..., 0::2, 1::2] = (x_ll + x_lh - x_hl - x_hh) / recon_divisor
    x[..., 1::2, 0::2] = (x_ll - x_lh + x_hl - x_hh) / recon_divisor
    x[..., 1::2, 1::2] = (x_ll - x_lh - x_hl + x_hh) / recon_divisor
    return x

def dwt_video(frames, haar_mode="ortho"):
    if frames.ndim != 5:
        raise ValueError(f"frames must have shape (B, T, C, H, W), got {tuple(frames.shape)}")
    B, T, C, H, W = frames.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"DWT requires even spatial size, got H={H}, W={W}")
    x = frames.reshape(B * T, C, H, W)
    x_ll, x_lh, x_hl, x_hh = dwt_2d(x, haar_mode=haar_mode)
    ll = x_ll.reshape(B, T, C, H // 2, W // 2)
    hf = torch.cat([x_lh, x_hl, x_hh], dim=1)
    hf = hf.reshape(B, T, 3 * C, H // 2, W // 2)
    return ll, hf

def idwt_video(ll, hf, haar_mode="ortho"):
    if ll.ndim != 5 or hf.ndim != 5:
        raise ValueError(
            f"ll and hf must both have shape (B, T, C, H, W), got {tuple(ll.shape)} and {tuple(hf.shape)}"
        )
    B, T, C, h2, w2 = ll.shape
    if hf.shape[:2] != (B, T):
        raise ValueError(f"hf must match ll batch/time dims, got {tuple(hf.shape[:2])} vs {(B, T)}")
    if hf.shape[2] != 3 * C:
        raise ValueError(f"hf must have {3 * C} channels when ll has {C}, got {hf.shape[2]}")
    if hf.shape[-2:] != (h2, w2):
        raise ValueError(f"hf spatial size must match ll, got {tuple(hf.shape[-2:])} vs {(h2, w2)}")
    hf_flat = hf.reshape(B * T, 3 * C, h2, w2)
    x_lh, x_hl, x_hh = hf_flat.chunk(3, dim=1)
    ll_flat = ll.reshape(B * T, C, h2, w2)
    out = idwt_2d(ll_flat, x_lh, x_hl, x_hh, haar_mode=haar_mode)
    return out.reshape(B, T, C, h2 * 2, w2 * 2)


# ---------------------------------------------------------------------------
#  Gamma schedule
# ---------------------------------------------------------------------------

def cosine_gamma_schedule(timesteps, s=8e-3, ratio=0.5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps + s
    thetas = torch.cos(t / (1 + s) * math.pi / 2).pow(2)
    thetas = thetas / thetas[0]
    thetas = thetas * ratio + (1.0 - ratio)
    gammas = 1.0 - (thetas[1:] / thetas[:-1])
    return torch.clip(gammas, 0, 0.999)

def linear_gamma_schedule(timesteps, linear_start=1e-4, linear_end=2e-2):
    return (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5,
                        timesteps, dtype=torch.float64) ** 2
    )


# ---------------------------------------------------------------------------
#  WaveletResidualDynamicalDiffusion
# ---------------------------------------------------------------------------

class WaveletResidualDynamicalDiffusion(GaussianDiffusion):
    def __init__(
        self, model, context_encoder, *, timesteps=1000, sampling_timesteps=None,
        objective='pred_noise', beta_schedule='sigmoid', schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0., auto_normalize=False, offset_noise_strength=0.,
        min_snr_loss_weight=False, min_snr_gamma=5,
        gamma_schedule='cosine', gamma_ratio=0.5,
    ):
        super().__init__(
            model, context_encoder, timesteps=timesteps, sampling_timesteps=sampling_timesteps,
            objective=objective, beta_schedule=beta_schedule, schedule_fn_kwargs=schedule_fn_kwargs,
            ddim_sampling_eta=ddim_sampling_eta, auto_normalize=auto_normalize,
            offset_noise_strength=offset_noise_strength, min_snr_loss_weight=min_snr_loss_weight,
            min_snr_gamma=min_snr_gamma,
        )

        if gamma_schedule == 'cosine':
            gammas = cosine_gamma_schedule(timesteps, ratio=gamma_ratio)
        elif gamma_schedule == 'linear':
            gammas = linear_gamma_schedule(timesteps)
        else:
            raise ValueError(f"Unknown gamma_schedule: {gamma_schedule}")

        thetas = 1.0 - gammas
        thetas_cumprod = torch.cumprod(thetas, dim=0)
        thetas_cumprod_prev = F.pad(thetas_cumprod[:-1], (1, 0), value=1.0)

        reg = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        reg('gammas',                        gammas)
        reg('thetas_cumprod',                thetas_cumprod)
        reg('thetas_cumprod_prev',           thetas_cumprod_prev)
        reg('sqrt_thetas_cumprod',           torch.sqrt(thetas_cumprod))
        reg('sqrt_one_minus_thetas_cumprod', torch.sqrt(1.0 - thetas_cumprod))

    def load_backbone(self, backbone_net):
        self.backbone_net = backbone_net


    @staticmethod
    def get_emas(x, sqrt_theta, sqrt_one_minus_theta):
        B, T, C, H, W = x.shape
        emas = [x[:, 0]]
        for s in range(1, T):
            emas.append(sqrt_one_minus_theta * emas[-1] + sqrt_theta * x[:, s])
        return torch.stack(emas, dim=1)

    @staticmethod
    def get_reverse_emas(x_ema, sqrt_theta, sqrt_one_minus_theta):
        B, T, C, H, W = x_ema.shape
        xs = [x_ema[:, 0]]
        for s in range(1, T):
            xs.append((x_ema[:, s] - sqrt_one_minus_theta * x_ema[:, s - 1]) / sqrt_theta)
        return torch.stack(xs, dim=1)

    def _theta_coeffs(self, t):
        b = t.shape[0]
        sq  = self.sqrt_thetas_cumprod.gather(-1, t).reshape(b, 1, 1, 1)
        sqm = self.sqrt_one_minus_thetas_cumprod.gather(-1, t).reshape(b, 1, 1, 1)
        return sq, sqm

    @staticmethod
    def _extract5d(a, t, shape_5d):
        return extract(a, t, shape_5d)

    # ----- Forward process -----

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        sqrt_theta, sqrt_1m_theta = self._theta_coeffs(t)
        noise_ema = self.get_emas(noise, sqrt_theta, sqrt_1m_theta)
        x_ema     = self.get_emas(x_start, sqrt_theta, sqrt_1m_theta)

        ref5d = x_start[:, :1, :1].shape
        sqrt_alpha    = self._extract5d(self.sqrt_alphas_cumprod, t, ref5d)
        sqrt_1m_alpha = self._extract5d(self.sqrt_one_minus_alphas_cumprod, t, ref5d)

        z_t = sqrt_alpha * x_ema + sqrt_1m_alpha * noise_ema
        return z_t, noise_ema

    # ----- Training loss -----

    def p_losses(self, x_start, t, cond=None, ctx=None, idx=None, noise=None, ll_mag=None,
                 intensity_weight=0.0, return_pred_x_start=False):
        noise = default(noise, lambda: torch.randn_like(x_start))
        z_t, noise_ema = self.q_sample(x_start, t, noise)
        model_output = self.model(z_t, t, cond=cond, ctx=ctx, idx=idx)
        sqrt_theta = sqrt_1m_theta = None

        if self.objective == 'pred_noise':
            target = noise_ema
            if return_pred_x_start:
                sqrt_theta, sqrt_1m_theta = self._theta_coeffs(t)
                x_start_ema = self.predict_start_from_noise(z_t, t, model_output).clamp(-1., 1.)
                pred_x_start = self.get_reverse_emas(x_start_ema, sqrt_theta, sqrt_1m_theta).clamp(-1., 1.)
        elif self.objective == 'pred_x0':
            sqrt_theta, sqrt_1m_theta = self._theta_coeffs(t)
            target = self.get_emas(x_start, sqrt_theta, sqrt_1m_theta)
            if return_pred_x_start:
                x_start_ema = model_output.clamp(-1., 1.)
                pred_x_start = self.get_reverse_emas(x_start_ema, sqrt_theta, sqrt_1m_theta).clamp(-1., 1.)
        elif self.objective == 'pred_v':
            sqrt_theta, sqrt_1m_theta = self._theta_coeffs(t)
            x_ema = self.get_emas(x_start, sqrt_theta, sqrt_1m_theta)
            target = self.predict_v(x_ema, t, noise_ema)
            if return_pred_x_start:
                x_start_ema = self.predict_start_from_v(z_t, t, model_output).clamp(-1., 1.)
                pred_x_start = self.get_reverse_emas(x_start_ema, sqrt_theta, sqrt_1m_theta).clamp(-1., 1.)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        loss = F.mse_loss(model_output, target, reduction='none')


        if intensity_weight > 0 and ll_mag is not None:
            intensity_w = 1.0 + intensity_weight * ll_mag
            loss = loss * intensity_w

        loss = loss.mean(dim=[1, 2, 3, 4])
        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss = loss.mean()
        if return_pred_x_start:
            return loss, pred_x_start.detach()
        return loss


    def model_predictions_dynamic(self, x, t, cond=None, ctx=None, idx=None,
                                  clip_x_start=False, sqrt_theta=None, sqrt_1m_theta=None):
        model_output = self.model(x, t, cond=cond, ctx=ctx, idx=idx)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise_ema = model_output
            x_start_ema = self.predict_start_from_noise(x, t, pred_noise_ema)
            x_start_ema = maybe_clip(x_start_ema)
            x_start = self.get_reverse_emas(x_start_ema, sqrt_theta, sqrt_1m_theta)
            x_start = maybe_clip(x_start)
            x_start_ema = self.get_emas(x_start, sqrt_theta, sqrt_1m_theta)
            pred_noise_ema = self.predict_noise_from_start(x, t, x_start_ema)

        elif self.objective == 'pred_x0':
            x_start_ema = model_output
            x_start_ema = maybe_clip(x_start_ema)
            x_start = self.get_reverse_emas(x_start_ema, sqrt_theta, sqrt_1m_theta)
            x_start = maybe_clip(x_start)
            x_start_ema = self.get_emas(x_start, sqrt_theta, sqrt_1m_theta)
            pred_noise_ema = self.predict_noise_from_start(x, t, x_start_ema)

        elif self.objective == 'pred_v':
            v = model_output
            x_start_ema = self.predict_start_from_v(x, t, v)
            x_start_ema = maybe_clip(x_start_ema)
            x_start = self.get_reverse_emas(x_start_ema, sqrt_theta, sqrt_1m_theta)
            x_start = maybe_clip(x_start)
            x_start_ema = self.get_emas(x_start, sqrt_theta, sqrt_1m_theta)
            pred_noise_ema = self.predict_noise_from_start(x, t, x_start_ema)

        return pred_noise_ema, x_start, x_start_ema

    def predict_start_from_noise(self, x_t, t, noise):
        ref5d = x_t[:, :1, :1].shape
        sa  = self._extract5d(self.sqrt_recip_alphas_cumprod, t, ref5d)
        sam = self._extract5d(self.sqrt_recipm1_alphas_cumprod, t, ref5d)
        return sa * x_t - sam * noise

    def predict_noise_from_start(self, x_t, t, x0):
        ref5d = x_t[:, :1, :1].shape
        sa  = self._extract5d(self.sqrt_recip_alphas_cumprod, t, ref5d)
        sam = self._extract5d(self.sqrt_recipm1_alphas_cumprod, t, ref5d)
        return (sa * x_t - x0) / sam

    def predict_v(self, x_start, t, noise):
        ref5d = x_start[:, :1, :1].shape
        sa  = self._extract5d(self.sqrt_alphas_cumprod, t, ref5d)
        sam = self._extract5d(self.sqrt_one_minus_alphas_cumprod, t, ref5d)
        return sa * noise - sam * x_start

    def predict_start_from_v(self, x_t, t, v):
        ref5d = x_t[:, :1, :1].shape
        sa  = self._extract5d(self.sqrt_alphas_cumprod, t, ref5d)
        sam = self._extract5d(self.sqrt_one_minus_alphas_cumprod, t, ref5d)
        return sa * x_t - sam * v

    # ----- DDIM Sampling -----

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, ctx=None, idx=None, return_all_timesteps=False):
        batch = shape[0]
        device = cond.device if cond is not None else self.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        z = torch.randn(shape, device=device)
        t_max = times[0]
        tc_max = torch.full((batch,), t_max, device=device, dtype=torch.long)
        sqrt_theta_max, sqrt_1m_theta_max = self._theta_coeffs(tc_max)
        z = self.get_emas(z, sqrt_theta_max, sqrt_1m_theta_max)

        imgs = [z]

        for time, time_next in tqdm(time_pairs, desc='WHIRLD DDIM', disable=True):
            tc = torch.full((batch,), time, device=device, dtype=torch.long)
            sqrt_theta_t, sqrt_1m_theta_t = self._theta_coeffs(tc)

            if time_next >= 0:
                tc_prev = torch.full((batch,), time_next, device=device, dtype=torch.long)
                sqrt_theta_prev, sqrt_1m_theta_prev = self._theta_coeffs(tc_prev)
            else:
                sqrt_theta_prev = torch.ones_like(sqrt_theta_t)
                sqrt_1m_theta_prev = torch.zeros_like(sqrt_1m_theta_t)

            pred_noise_ema, x_start, x_start_ema = self.model_predictions_dynamic(
                z, tc, cond=cond, ctx=ctx, idx=idx, clip_x_start=True,
                sqrt_theta=sqrt_theta_t, sqrt_1m_theta=sqrt_1m_theta_t,
            )

            if time_next < 0:
                z = x_start
                imgs.append(z)
                continue

            x_start_ema_prev = self.get_emas(x_start, sqrt_theta_prev, sqrt_1m_theta_prev)
            noise_per_frame = self.get_reverse_emas(pred_noise_ema, sqrt_theta_t, sqrt_1m_theta_t)
            noise_ema_prev  = self.get_emas(noise_per_frame, sqrt_theta_prev, sqrt_1m_theta_prev)

            alpha      = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c     = (1 - alpha_next - sigma ** 2).sqrt()

            stochastic_noise = torch.randn_like(z)
            stochastic_noise_ema = self.get_emas(stochastic_noise, sqrt_theta_prev, sqrt_1m_theta_prev)

            z = x_start_ema_prev * alpha_next.sqrt() + c * noise_ema_prev + sigma * stochastic_noise_ema
            imgs.append(z)

        return z if not return_all_timesteps else torch.stack(imgs, dim=1)


    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, ctx=None, idx=None, return_all_timesteps=False):
        batch, device = shape[0], cond.device if cond is not None else self.device

        z = torch.randn(shape, device=device)
        t_max = self.num_timesteps - 1
        tc_max = torch.full((batch,), t_max, device=device, dtype=torch.long)
        sqrt_theta_max, sqrt_1m_theta_max = self._theta_coeffs(tc_max)
        z = self.get_emas(z, sqrt_theta_max, sqrt_1m_theta_max)

        imgs = [z]

        for t_int in tqdm(reversed(range(0, self.num_timesteps)), desc='WHIRLD sampling',
                          total=self.num_timesteps, disable=True):
            tc = torch.full((batch,), t_int, device=device, dtype=torch.long)
            sqrt_theta_t, sqrt_1m_theta_t = self._theta_coeffs(tc)

            pred_noise_ema, x_start, _ = self.model_predictions_dynamic(
                z, tc, cond=cond, ctx=ctx, idx=idx, clip_x_start=True,
                sqrt_theta=sqrt_theta_t, sqrt_1m_theta=sqrt_1m_theta_t,
            )

            if t_int > 0:
                tc_prev = torch.full((batch,), t_int - 1, device=device, dtype=torch.long)
                sqrt_theta_prev, sqrt_1m_theta_prev = self._theta_coeffs(tc_prev)
            else:
                sqrt_theta_prev = torch.ones_like(sqrt_theta_t)
                sqrt_1m_theta_prev = torch.zeros_like(sqrt_1m_theta_t)

            x_start_ema_prev = self.get_emas(x_start, sqrt_theta_prev, sqrt_1m_theta_prev)

            model_mean = (
                extract(self.posterior_mean_coef1, tc, z.shape) * x_start_ema_prev +
                extract(self.posterior_mean_coef2, tc, z.shape) * z
            )
            model_log_var = extract(self.posterior_log_variance_clipped, tc, z.shape)

            noise = torch.randn_like(z) if t_int > 0 else 0.
            z = model_mean + (0.5 * model_log_var).exp() * noise
            imgs.append(z)

        return z if not return_all_timesteps else torch.stack(imgs, dim=1)

    # ----- Fragment-level interfaces -----

    def train_hf_fragment(self, hf_target, t, cond_hf, ctx, idx, ll_mag=None,
                          intensity_weight=0.0, return_pred_x_start=False):
        return self.p_losses(hf_target, t, cond=cond_hf, ctx=ctx, idx=idx, 
                             ll_mag=ll_mag, intensity_weight=intensity_weight,
                             return_pred_x_start=return_pred_x_start)

    @torch.no_grad()
    def sample_hf_fragment(self, shape, cond, ctx, idx, return_all_timesteps=False):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(shape, cond=cond, ctx=ctx, idx=idx, return_all_timesteps=return_all_timesteps)


# ---------------------------------------------------------------------------
#  WHIRLD
# ---------------------------------------------------------------------------

class WHIRLD(nn.Module):
    def __init__(self, diffusion, lambda_diff=0.5, residual_scale=2.0, 
                 noise_floor=0.01, n_ensemble=1, intensity_weight=0.0,
                 weak_echo_mode="soft", weak_echo_temperature=0.02,
                 intensity_log_scale=4.0,
                 haar_mode="ortho", history_blend_alpha=0.15,
                 history_noise_std=0.05):
        super().__init__()
        self.diffusion = diffusion
        if diffusion.model.frame_channels % 3 != 0:
            raise ValueError(
                f"Expected HF model channels to be divisible by 3, got {diffusion.model.frame_channels}"
            )
        
        self.lambda_diff = lambda_diff
        self.n_ensemble  = n_ensemble
        self.intensity_weight = intensity_weight
        self.noise_floor = noise_floor
        self.weak_echo_mode = weak_echo_mode
        self.weak_echo_temperature = weak_echo_temperature
        self.intensity_log_scale = intensity_log_scale
        self.haar_mode = haar_mode
        self.history_blend_alpha = history_blend_alpha
        self.history_noise_std = history_noise_std
        self.img_channels = diffusion.model.frame_channels // 3
        self.t_in = diffusion.model.T_in
        self.min_spatial_multiple = 2 ** len(diffusion.context_encoder.downs)
        if self.weak_echo_mode not in {"soft", "hard", "none"}:
            raise ValueError(
                f"weak_echo_mode must be one of 'soft', 'hard', or 'none', got {self.weak_echo_mode}"
            )
        if self.weak_echo_temperature <= 0:
            raise ValueError(f"weak_echo_temperature must be positive, got {self.weak_echo_temperature}")
        if self.intensity_log_scale <= 0:
            raise ValueError(f"intensity_log_scale must be positive, got {self.intensity_log_scale}")
        if self.haar_mode not in {"legacy", "ortho"}:
            raise ValueError(f"haar_mode must be 'legacy' or 'ortho', got {self.haar_mode}")
        if not (0.0 <= self.history_blend_alpha <= 1.0):
            raise ValueError(f"history_blend_alpha must be in [0, 1], got {self.history_blend_alpha}")
        if self.history_noise_std < 0:
            raise ValueError(f"history_noise_std must be non-negative, got {self.history_noise_std}")

        # Moderate scaling to boost HF residual SNR
        self.residual_scale = residual_scale
        self.scale   = lambda x: x * self.residual_scale
        self.unscale = lambda x: x / self.residual_scale

        print(f"[WHIRLD] lambda_diff={lambda_diff}, residual_scale={residual_scale}, "
              f"noise_floor={noise_floor}, weak_echo_mode={weak_echo_mode}, "
              f"n_ensemble={n_ensemble}, intensity_weight={intensity_weight}, "
              f"intensity_log_scale={intensity_log_scale}, haar_mode={haar_mode}, "
              f"history_blend_alpha={history_blend_alpha}, history_noise_std={history_noise_std}")

    @property
    def device(self):
        return self.diffusion.betas.device

    def load_backbone(self, backbone_net):
        self.diffusion.load_backbone(backbone_net)
        self.backbone_net = backbone_net
        self.default_t_out = getattr(backbone_net, "T_out", None)

    def _validate_video_tensor(self, frames, name, expected_t=None):
        if not torch.is_tensor(frames):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(frames).__name__}")
        if frames.ndim != 5:
            raise ValueError(f"{name} must have shape (B, T, C, H, W), got {tuple(frames.shape)}")
        B, T, C, H, W = frames.shape
        if expected_t is not None and T != expected_t:
            raise ValueError(f"{name} must contain {expected_t} frames, got {T}")
        if C != self.img_channels:
            raise ValueError(f"{name} must have {self.img_channels} channel(s), got {C}")
        if H % self.min_spatial_multiple != 0 or W % self.min_spatial_multiple != 0:
            raise ValueError(
                f"{name} spatial size must be divisible by {self.min_spatial_multiple}, got H={H}, W={W}"
            )
        if not torch.isfinite(frames).all():
            raise ValueError(f"{name} contains NaN or Inf values")
        return B, T, C, H, W

    def _validate_horizon(self, T_out):
        if T_out <= 0:
            raise ValueError(f"T_out must be positive, got {T_out}")
        if T_out % self.t_in != 0:
            raise ValueError(f"T_out ({T_out}) must be divisible by T_in ({self.t_in}) for fragment autoregression.")

    def _validate_backbone_output(self, backbone_output, B, T_out, C, H, W):
        expected_shape = (B, T_out, C, H, W)
        if backbone_output.shape != expected_shape:
            raise ValueError(
                f"backbone_output must have shape {expected_shape}, got {tuple(backbone_output.shape)}"
            )
        if not torch.isfinite(backbone_output).all():
            raise ValueError("backbone_output contains NaN or Inf values")

    def _scan_ctx_chunk(self, ctx_chunk, start_state=None):
        if ctx_chunk.ndim != 5:
            raise ValueError(f"ctx_chunk must have shape (B, T, C, H, W), got {tuple(ctx_chunk.shape)}")
        B, T, C, h2, w2 = ctx_chunk.shape
        if start_state is None:
            self.diffusion.context_encoder.init_state((B, C, h2, w2), ctx_chunk.device)
        else:
            self.diffusion.context_encoder.set_state(start_state)

        last_ctx = None
        for i in range(T):
            last_ctx = self.diffusion.context_encoder.forward(ctx_chunk[:, i])

        if last_ctx is None:
            raise ValueError("ctx_chunk must contain at least one frame")
        return [h.clone() for h in last_ctx], self.diffusion.context_encoder.get_state()

    def _init_past_ctx_state(self, ll_in_s, hf_in_s):
        past_ctx = torch.cat((ll_in_s, hf_in_s), dim=2)
        return self._scan_ctx_chunk(past_ctx)

    def _suppress_weak_echoes(self, frames_pred):
        if self.noise_floor <= 0 or self.weak_echo_mode == "none":
            return frames_pred
        if self.weak_echo_mode == "hard":
            return torch.where(
                frames_pred < self.noise_floor,
                torch.zeros_like(frames_pred),
                frames_pred,
            )

        gate = torch.sigmoid((frames_pred - self.noise_floor) / self.weak_echo_temperature)
        return frames_pred * gate

    def _build_intensity_map(self, ll_mag):
        if ll_mag is None:
            return None
        ll_mag = ll_mag.clamp_min(0)
        ll_peak = ll_mag.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        ll_norm = (ll_mag / ll_peak).clamp_(0, 1)
        return torch.log1p(self.intensity_log_scale * ll_norm) / math.log1p(self.intensity_log_scale)

    def _make_training_history(self, teacher_hf_frag, hf_mu):
        residual = (teacher_hf_frag - hf_mu).detach()
        if self.history_blend_alpha <= 0 and self.history_noise_std <= 0:
            return teacher_hf_frag.detach()

        batch = residual.shape[0]
        atten = 1.0 - self.history_blend_alpha * torch.rand(
            (batch, 1, 1, 1, 1), device=residual.device, dtype=residual.dtype
        )
        residual_rms = residual.pow(2).mean(dim=(2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
        noise = torch.randn_like(residual) * residual_rms * self.history_noise_std
        return hf_mu + residual * atten + noise

    def predict(self, frames_in, compute_loss=False, frames_gt=None, **kwargs):
        T_out = default(
            kwargs.get('T_out'),
            frames_gt.shape[1] if frames_gt is not None else self.default_t_out
        )
        if T_out is None:
            raise ValueError("T_out must be provided for sampling when backbone output horizon is unavailable")
        if compute_loss and frames_gt is None:
            raise ValueError("frames_gt must be provided when compute_loss=True")
        if compute_loss and frames_gt is not None:
            return self._train_forward(frames_in, frames_gt, T_out)
        else:
            return self._sample_forward(frames_in, T_out)

    # ===================== Training =====================

    def _train_forward(self, frames_in, frames_gt, T_out):
        if not hasattr(self, 'backbone_net'):
            raise RuntimeError("backbone_net is not loaded. Call load_backbone() before training.")

        B, T_in, C, H, W = self._validate_video_tensor(frames_in, "frames_in", expected_t=self.t_in)
        self._validate_horizon(T_out)
        gt_shape = self._validate_video_tensor(frames_gt, "frames_gt", expected_t=T_out)
        if (gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]) != (B, C, H, W):
            raise ValueError(
                "frames_gt must match frames_in in batch, channel, and spatial dimensions"
            )
        device = self.device

        # 1. Backbone prediction
        backbone_output, backbone_loss = self.backbone_net.predict(
            frames_in, frames_gt=frames_gt, compute_loss=True
        )
        backbone_output = backbone_output.detach().clamp(0, 1)
        self._validate_backbone_output(backbone_output, B, T_out, C, H, W)

        # 2. Wavelet decomposition
        ll_gt,   hf_gt       = dwt_video(frames_gt, haar_mode=self.haar_mode)
        ll_pred, hf_backbone = dwt_video(backbone_output, haar_mode=self.haar_mode)
        ll_in,   hf_in       = dwt_video(frames_in, haar_mode=self.haar_mode)

        # 3. Construct the high-frequency residual target.
        hf_res = self.scale(hf_gt - hf_backbone)  # (B, T_out, 3C, H/2, W/2)

        # 4. Build context from observed past coefficients and deterministic
        #    future coefficients.
        ll_in_s  = self.scale(ll_in)
        hf_in_s  = self.scale(hf_in)
        ll_pred_s = self.scale(ll_pred)
        hf_backbone_s = self.scale(hf_backbone)
        # 5. Fragment-wise training
        total_loss = 0.0
        pre_hf_frag = hf_in_s   # (B, T_in, 3C, H/2, W/2)
        pre_hf_mu = None
        n_frags = T_out // T_in
        _, prefix_state = self._init_past_ctx_state(ll_in_s, hf_in_s)

        # For intensity weighting, extract magnitude of target LL
        ll_gt_mag = ll_gt.abs().mean(dim=2, keepdim=True)

        for frag_idx in range(n_frags):
            frag_start = frag_idx * T_in
            frag_end   = frag_start + T_in

            hf_mu = hf_backbone_s[:, frag_start:frag_end]
            target_hf_frag = hf_res[:, frag_start:frag_end]
            ll_mag_frag = self._build_intensity_map(ll_gt_mag[:, frag_start:frag_end])

            # Autoregressive condition: Previous HF fragment relative to its mean
            cond_hf = (pre_hf_frag - pre_hf_mu 
                       if pre_hf_mu is not None 
                       else torch.zeros_like(pre_hf_frag))

            t   = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device).long()
            ctx_chunk = torch.cat((ll_pred_s[:, frag_start:frag_end], hf_mu), dim=2)
            ctx, _ = self._scan_ctx_chunk(ctx_chunk, start_state=prefix_state)
            idx = torch.full((B,), frag_idx, device=device, dtype=torch.long)

            frag_loss = self.diffusion.train_hf_fragment(
                target_hf_frag, t, cond_hf, ctx, idx,
                ll_mag=ll_mag_frag, intensity_weight=self.intensity_weight
            )
            total_loss = total_loss + frag_loss

            # Strict Wavelet-domain update
            teacher_hf_frag = target_hf_frag + hf_mu
            pre_hf_frag = self._make_training_history(teacher_hf_frag, hf_mu)
            pre_hf_mu = hf_mu
            writeback_chunk = torch.cat((ll_pred_s[:, frag_start:frag_end], pre_hf_frag), dim=2)
            _, prefix_state = self._scan_ctx_chunk(writeback_chunk, start_state=prefix_state)

        diff_loss = total_loss / n_frags

        if backbone_loss is not None:
            loss = backbone_loss + self.lambda_diff * diff_loss
        else:
            loss = diff_loss

        return None, {
            'total_loss': loss,
            'backbone_loss': backbone_loss.detach() if backbone_loss is not None else loss.detach() * 0,
            'diff_loss': diff_loss.detach(),
        }

    # ===================== Sampling =====================

    def _run_diffusion_chain(self, ll_in_s, hf_in_s, ll_pred_s, hf_backbone_s,
                             B, T_in, hf_c, h2, w2, n_frags, T_out, device):
        """Generate the three high-frequency residual subbands autoregressively."""
        hf_residuals = []
        pre_hf_frag = hf_in_s
        pre_hf_mu = None
        _, prefix_state = self._init_past_ctx_state(ll_in_s, hf_in_s)

        for frag_idx in range(n_frags):
            frag_start = frag_idx * T_in
            frag_end = frag_start + T_in
            hf_mu = hf_backbone_s[:, frag_idx * T_in:(frag_idx + 1) * T_in]

            cond_hf = (pre_hf_frag - pre_hf_mu 
                       if pre_hf_mu is not None 
                       else torch.zeros_like(pre_hf_frag))
                       
            ctx_chunk = torch.cat((ll_pred_s[:, frag_start:frag_end], hf_mu), dim=2)
            ctx, _ = self._scan_ctx_chunk(ctx_chunk, start_state=prefix_state)
            idx = torch.full((B,), frag_idx, device=device, dtype=torch.long)

            # Sample 3C HF residual
            hf_residual = self.diffusion.sample_hf_fragment(
                (B, T_in, hf_c, h2, w2), cond=cond_hf, ctx=ctx, idx=idx,
            )
            
            hf_residuals.append(hf_residual)

            # Update the autoregressive condition in the wavelet domain.
            pre_hf_frag = hf_residual + hf_mu
            pre_hf_mu = hf_mu
            writeback_chunk = torch.cat((ll_pred_s[:, frag_start:frag_end], pre_hf_frag), dim=2)
            _, prefix_state = self._scan_ctx_chunk(writeback_chunk, start_state=prefix_state)

        return torch.cat(hf_residuals, dim=1)

    @torch.no_grad()
    def _sample_forward(self, frames_in, T_out):
        if not hasattr(self, 'backbone_net'):
            raise RuntimeError("backbone_net is not loaded. Call load_backbone() before sampling.")

        B, T_in, C, H, W = self._validate_video_tensor(frames_in, "frames_in", expected_t=self.t_in)
        self._validate_horizon(T_out)
        device = self.device

        backbone_output, _ = self.backbone_net.predict(frames_in)
        backbone_output = backbone_output.clamp(0, 1)
        self._validate_backbone_output(backbone_output, B, T_out, C, H, W)

        ll_pred, hf_backbone = dwt_video(backbone_output, haar_mode=self.haar_mode)
        ll_in,   hf_in       = dwt_video(frames_in, haar_mode=self.haar_mode)

        ll_in_s  = self.scale(ll_in)
        hf_in_s  = self.scale(hf_in)
        ll_pred_s = self.scale(ll_pred)
        hf_backbone_s = self.scale(hf_backbone)

        hf_c = 3 * C
        h2, w2 = ll_in.shape[3], ll_in.shape[4]
        n_frags = T_out // T_in

        chain_args = (ll_in_s, hf_in_s, ll_pred_s, hf_backbone_s,
                      B, T_in, hf_c, h2, w2, n_frags, T_out, device)

        # Ensemble Sampling on High-Frequencies
        if self.n_ensemble > 1:
            preds = [self._run_diffusion_chain(*chain_args) for _ in range(self.n_ensemble)]
            hf_res_all = torch.stack(preds).mean(dim=0)
        else:
            hf_res_all = self._run_diffusion_chain(*chain_args)

        # Restore the residual scale and reconstruct with the retained LL component.
        hf_pred = hf_backbone + self.unscale(hf_res_all)
        frames_pred = idwt_video(ll_pred, hf_pred, haar_mode=self.haar_mode).clamp(0, 1)

        frames_pred = self._suppress_weak_echoes(frames_pred)

        return frames_pred, None


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def get_model(
    img_channels=1,
    dim=96,
    dim_mults=(1, 2, 4, 8),
    T_in=5,
    T_out=20,
    timesteps=1000,
    sampling_timesteps=250,
    gamma_schedule='cosine',
    gamma_ratio=0.5,
    objective='pred_noise',
    lambda_diff=0.5,       
    residual_scale=2.0,    
    noise_floor=0.01,      
    n_ensemble=1,          
    intensity_weight=3.0,  
    weak_echo_mode="soft",
    weak_echo_temperature=0.02,
    intensity_log_scale=4.0,
    haar_mode="ortho",
    history_blend_alpha=0.15,
    history_noise_std=0.05,
    **kwargs,
):
    if dim % 3 != 0:
        raise ValueError(f"dim must be divisible by 3 when grouped HF convolutions are enabled, got {dim}")

    # DWUNet processes the three directional high-frequency subbands.
    hf_channels = 3 * img_channels

    # Context frames contain one low-frequency and three high-frequency subbands.
    context_channels = 4 * img_channels

    denoiser = DWUNet(
        dim=dim,
        T_in=T_in,
        frame_channels=hf_channels,
        dim_mults=dim_mults,
    )
    
    context_encoder = WaveletContextEncoder(
        dim=dim,
        dim_mults=dim_mults,
        channels=context_channels,
    )

    diffusion = WaveletResidualDynamicalDiffusion(
        model=denoiser,
        context_encoder=context_encoder,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        gamma_schedule=gamma_schedule,
        gamma_ratio=gamma_ratio,
        objective=objective,
        auto_normalize=False,    
    )

    whirld = WHIRLD(
        diffusion,
        lambda_diff=lambda_diff,
        residual_scale=residual_scale,
        noise_floor=noise_floor,
        n_ensemble=n_ensemble,
        intensity_weight=intensity_weight,
        weak_echo_mode=weak_echo_mode,
        weak_echo_temperature=weak_echo_temperature,
        intensity_log_scale=intensity_log_scale,
        haar_mode=haar_mode,
        history_blend_alpha=history_blend_alpha,
        history_noise_std=history_noise_std,
    )
    return whirld


__all__ = [
    "WHIRLD",
    "DWUNet",
    "WaveletContextEncoder",
    "WaveletResidualDynamicalDiffusion",
    "get_model",
]
