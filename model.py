import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.transpose(-1, -2).unbind(dim=1)  # each: (b, heads, hw, d)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Block(nn.Module):
    """ResBlock, optionally followed by self-attention."""

    def __init__(self, in_ch, out_ch, time_dim, dropout, use_attn):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, temb):
        return self.attn(self.res(x, temb))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, img_ch=3, ch=64, ch_mult=(1, 2, 2, 2), num_res=2,
                 attn_res=(16,), dropout=0.1, img_size=32):
        super().__init__()
        tdim = ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(ch), nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim)
        )
        self.stem = nn.Conv2d(img_ch, ch, 3, padding=1)

        # Encoder: record channel counts of every activation saved for skip connections
        self.down = nn.ModuleList()
        skip_chs, cur, res = [ch], ch, img_size
        for i, m in enumerate(ch_mult):
            out = ch * m
            for _ in range(num_res):
                self.down.append(Block(cur, out, tdim, dropout, res in attn_res))
                cur = out
                skip_chs.append(cur)
            if i != len(ch_mult) - 1:
                self.down.append(Downsample(cur))
                skip_chs.append(cur)
                res //= 2

        self.mid = nn.ModuleList([
            Block(cur, cur, tdim, dropout, use_attn=True),
            Block(cur, cur, tdim, dropout, use_attn=False),
        ])

        # Decoder: each block consumes one skip connection
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(ch_mult))):
            out = ch * m
            for _ in range(num_res + 1):
                self.up.append(Block(cur + skip_chs.pop(), out, tdim, dropout, res in attn_res))
                cur = out
            if i != 0:
                self.up.append(Upsample(cur))
                res *= 2

        self.out = nn.Sequential(nn.GroupNorm(32, cur), nn.SiLU(), nn.Conv2d(cur, img_ch, 3, padding=1))

    def forward(self, x, t):
        temb = self.time_mlp(t)
        h = self.stem(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, temb) if isinstance(layer, Block) else layer(h)
            skips.append(h)
        for layer in self.mid:
            h = layer(h, temb)
        for layer in self.up:
            if isinstance(layer, Block):
                h = layer(torch.cat([h, skips.pop()], dim=1), temb)
            else:
                h = layer(h)
        return self.out(h)


def build_model(cfg):
    return UNet(
        ch=cfg["ch"],
        ch_mult=tuple(int(m) for m in cfg["ch_mult"].split(",")),
        num_res=cfg["num_res"],
        dropout=cfg["dropout"],
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6
