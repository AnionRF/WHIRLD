import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        upsampling=False,
        act_norm=False,
        act_inplace=True,
    ):
        super().__init__()
        self.act_norm = act_norm
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        downsampling=False,
        upsampling=False,
        act_norm=True,
        act_inplace=True,
    ):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            upsampling=upsampling,
            padding=padding,
            act_norm=act_norm,
            act_inplace=act_inplace,
        )

    def forward(self, x):
        return self.conv(x)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class Heat2D(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(n=224, device=torch.device("cpu"), dtype=torch.float):
        weight_x = (torch.linspace(0, n - 1, n, device=device, dtype=dtype).view(1, -1) + 0.5) / n
        weight_n = torch.linspace(0, n - 1, n, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / n)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        return torch.exp(-weight)

    def forward(self, x, freq_embed=None):
        b, c, h, w = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)

        if ((h, w) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_cosn = self.get_cos_map(h, device=x.device).detach_()
            weight_cosm = self.get_cos_map(w, device=x.device).detach_()
            weight_exp = self.get_decay_map((h, w), device=x.device).detach_()
            setattr(self, "__RES__", (h, w))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        n, m = weight_cosn.shape[0], weight_cosm.shape[0]
        x = F.conv1d(x.contiguous().view(b, h, -1), weight_cosn.contiguous().view(n, h, 1))
        x = F.conv1d(x.contiguous().view(-1, w, c), weight_cosm.contiguous().view(m, w, 1)).contiguous().view(b, n, m, -1)

        if self.infer_mode:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp)
        else:
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
            x = torch.einsum("bnmc,nmc->bnmc", x, weight_exp)

        x = F.conv1d(x.contiguous().view(b, n, -1), weight_cosn.t().contiguous().view(h, n, 1))
        x = F.conv1d(x.contiguous().view(-1, m, c), weight_cosm.t().contiguous().view(w, m, 1)).contiguous().view(b, h, w, -1)

        x = self.out_norm(x)
        x = x * nn.functional.silu(z)
        x = self.out_linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class GatedHeatBlock(nn.Module):
    def __init__(
        self,
        dim,
        res=14,
        expand_ratio=1.0,
        local_conv_kernel_size=3,
        drop_path=0.0,
        norm_layer=LayerNorm2d,
    ):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.in_proj = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, stride=1, padding=0)
        self.local_conv_main = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=local_conv_kernel_size,
            stride=1,
            padding="same",
            groups=hidden_dim,
        )
        self.local_conv_gate = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=local_conv_kernel_size,
            stride=1,
            padding="same",
            groups=hidden_dim,
        )
        self.mixer = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim)
        self.act = nn.SiLU(inplace=True)
        self.norm = norm_layer(hidden_dim)
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, freq_embed=None):
        shortcut = x
        xz = self.in_proj(x)
        x_main, z_gate = xz.chunk(2, dim=1)
        x_main = self.local_conv_main(x_main)
        x_main = self.act(x_main)
        x_main = self.mixer(x_main, freq_embed)
        x_main = self.norm(x_main)
        z_gate = self.local_conv_gate(z_gate)
        z_gate = self.act(z_gate)
        out = x_main * z_gate
        out = self.out_proj(out)
        return shortcut + self.drop_path(out)


class TemporalAttentionModule(nn.Module):
    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False),
            nn.Sigmoid(),
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


class TemporalAttention(nn.Module):
    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        shortcut = x.clone() if self.attn_shortcut else None
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class MixMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TAUBranch(nn.Module):
    def __init__(self, dim, kernel_size=21, mlp_ratio=4.0, drop=0.0, drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = TemporalAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"layer_scale_1", "layer_scale_2"}

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class GatedHeatBranch(nn.Module):
    def __init__(self, dim, input_resolution, expand_ratio=1.0, drop_path=0.0):
        super().__init__()
        res = input_resolution[0] if isinstance(input_resolution, (tuple, list)) else input_resolution
        self.block = GatedHeatBlock(dim=dim, res=res, expand_ratio=expand_ratio, drop_path=drop_path)
        hidden_dim = int(dim * expand_ratio)
        self.freq_embed = nn.Parameter(torch.zeros(res, res, hidden_dim))
        trunc_normal_(self.freq_embed, std=0.02)

    def forward(self, x):
        return self.block(x, self.freq_embed)


class HTBlock(nn.Module):
    def __init__(self, dim, input_resolution, mlp_ratio=8.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.heat = GatedHeatBranch(dim, input_resolution=input_resolution, expand_ratio=1.0, drop_path=drop_path)
        self.tau = TAUBranch(dim, kernel_size=21, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        gate_hidden = max(dim // 4, 4)
        self.router = nn.Sequential(
            nn.Conv2d(dim * 3, gate_hidden, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 1, kernel_size=3, stride=1, padding=1),
        )
        nn.init.constant_(self.router[-1].bias, -1.0)

    def forward(self, x):
        heat_out = self.heat(x)
        tau_out = self.tau(x)
        gate = torch.sigmoid(self.router(torch.cat([x, heat_out, tau_out], dim=1)))
        return heat_out + gate * (tau_out - heat_out)


def _sampling_pattern(n, reverse=False):
    samplings = [False, True] * (n // 2)
    return list(reversed(samplings[:n])) if reverse else samplings[:n]


class SpatialEncoder(nn.Module):
    def __init__(self, c_in, c_hid, n_s, spatio_kernel, act_inplace=True):
        super().__init__()
        samplings = _sampling_pattern(n_s)
        self.enc = nn.Sequential(
            ConvSC(c_in, c_hid, spatio_kernel, downsampling=samplings[0], act_inplace=act_inplace),
            *[ConvSC(c_hid, c_hid, spatio_kernel, downsampling=s, act_inplace=act_inplace) for s in samplings[1:]],
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class SpatialDecoder(nn.Module):
    def __init__(self, c_hid, c_out, n_s, spatio_kernel, act_inplace=True):
        super().__init__()
        samplings = _sampling_pattern(n_s, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(c_hid, c_hid, spatio_kernel, upsampling=s, act_inplace=act_inplace) for s in samplings[:-1]],
            ConvSC(c_hid, c_hid, spatio_kernel, upsampling=samplings[-1], act_inplace=act_inplace),
        )
        self.readout = nn.Conv2d(c_hid, c_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        y = self.dec[-1](hid + enc1)
        return self.readout(y)


class TranslatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None, mlp_ratio=8.0, drop=0.0, drop_path=0.0, layer_i=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else "heat_tau"

        if model_type == "tau":
            self.block = TAUBranch(
                in_channels,
                kernel_size=21,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                act_layer=nn.GELU,
            )
        elif model_type == "heat":
            self.block = GatedHeatBranch(
                in_channels,
                input_resolution=input_resolution,
                expand_ratio=1.0,
                drop_path=drop_path,
            )
        elif model_type == "heat_tau":
            self.block = HTBlock(
                in_channels,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
            )
        else:
            raise ValueError(f"SimHT only supports model_type in {{'tau', 'heat', 'heat_tau'}}, got {model_type}")

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class SpatiotemporalTranslator(nn.Module):
    def __init__(self, channel_in, channel_hid, n2, input_resolution=None, model_type=None, mlp_ratio=4.0, drop=0.0, drop_path=0.1):
        super().__init__()
        assert n2 >= 2 and mlp_ratio > 1
        self.n2 = n2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.n2)]

        enc_layers = [TranslatorBlock(channel_in, channel_hid, input_resolution, model_type, mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        for i in range(1, n2 - 1):
            enc_layers.append(TranslatorBlock(channel_hid, channel_hid, input_resolution, model_type, mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        enc_layers.append(TranslatorBlock(channel_hid, channel_in, input_resolution, model_type, mlp_ratio, drop, drop_path=drop_path, layer_i=n2 - 1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b, t * c, h, w)
        z = x
        for i in range(self.n2):
            z = self.enc[i](z)
        return z.reshape(b, t, c, h, w)


class SimHTCore(nn.Module):
    def __init__(
        self,
        in_shape,
        hid_S=16,
        hid_T=256,
        N_S=4,
        N_T=4,
        model_type="heat_tau",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        act_inplace=True,
        **kwargs,
    ):
        super().__init__()
        t, c, h, w = in_shape
        h, w = int(h / 2 ** (N_S / 2)), int(w / 2 ** (N_S / 2))
        act_inplace = False

        self.encoder = SpatialEncoder(c, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.decoder = SpatialDecoder(hid_S, c, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        self.translator = SpatiotemporalTranslator(
            t * hid_S,
            hid_T,
            N_T,
            input_resolution=(h, w),
            model_type=model_type,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
        )

    def forward(self, x_raw, **kwargs):
        b, t, c, h, w = x_raw.shape
        x = x_raw.reshape(b * t, c, h, w)
        embed, skip = self.encoder(x)
        _, c_hid, h_hid, w_hid = embed.shape
        z = embed.view(b, t, c_hid, h_hid, w_hid)
        hid = self.translator(z)
        hid = hid.reshape(b * t, c_hid, h_hid, w_hid)
        y = self.decoder(hid, skip)
        return y.reshape(b, t, c, h, w)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        legacy_modules = {
            "enc.": "encoder.",
            "dec.": "decoder.",
            "hid.": "translator.",
        }
        for old_name, new_name in legacy_modules.items():
            old_prefix = prefix + old_name
            for key in [k for k in state_dict if k.startswith(old_prefix)]:
                state_dict[prefix + new_name + key[len(old_prefix):]] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


class SimHT(nn.Module):
    def __init__(
        self,
        in_shape,
        T_in,
        T_out,
        hid_S=64,
        hid_T=256,
        N_S=2,
        N_T=6,
        model_type="heat_tau",
        mlp_ratio=8.0,
        drop=0.0,
        drop_path=0.0,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        **kwargs,
    ):
        super().__init__()
        c, h, w = in_shape
        self.T_in = T_in
        self.T_out = T_out
        self.core = SimHTCore(
            in_shape=(T_in, c, h, w),
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            model_type=model_type,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x_raw, **kwargs):
        return self.core(x_raw, **kwargs)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        legacy_prefix = prefix + "model."
        for key in [k for k in state_dict if k.startswith(legacy_prefix)]:
            state_dict[prefix + "core." + key[len(legacy_prefix):]] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def predict(self, frames_in, frames_gt=None, compute_loss=False, **kwargs):
        frames_pred = []
        cur_seq = frames_in.clone()
        for _ in range(self.T_out // self.T_in):
            cur_seq = self.forward(cur_seq)
            frames_pred.append(cur_seq)
        frames_pred = torch.cat(frames_pred, dim=1)
        loss = self.loss_fn(frames_pred, frames_gt) if compute_loss else None
        return frames_pred, loss


def get_model(in_shape, T_in, T_out, **kwargs):
    return SimHT(in_shape, T_in=T_in, T_out=T_out, **kwargs)


__all__ = ["SimHT", "SimHTCore", "HTBlock", "get_model"]
