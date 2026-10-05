from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=4, width=24):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(in_channels, width), ConvBlock(width, 2*width), ConvBlock(2*width, 4*width)
        self.pool = nn.AvgPool2d(2)
        self.d2, self.d1 = ConvBlock(6*width, 2*width), ConvBlock(3*width, width)
        self.out = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); b = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return self.out(d1)


class SpectralConv2d(nn.Module):
    def __init__(self, cin: int, cout: int, modes_y=12, modes_x=20):
        super().__init__()
        self.modes_y, self.modes_x = modes_y, modes_x
        scale = 1 / (cin * cout)
        self.weight = nn.Parameter(scale * torch.randn(cin, cout, modes_y, modes_x, dtype=torch.cfloat))

    def forward(self, x):
        ft = torch.fft.rfft2(x)
        out = torch.zeros(x.shape[0], self.weight.shape[1], x.shape[-2], x.shape[-1]//2+1, dtype=torch.cfloat, device=x.device)
        my, mx = min(self.modes_y, x.shape[-2]), min(self.modes_x, ft.shape[-1])
        out[:, :, :my, :mx] = torch.einsum("bixy,ioxy->boxy", ft[:, :, :my, :mx], self.weight[:, :, :my, :mx])
        return torch.fft.irfft2(out, s=x.shape[-2:])


class FNO2d(nn.Module):
    def __init__(self, in_channels=10, out_channels=4, width=32, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 2*width, 1), nn.GELU(), nn.Conv2d(2*width, out_channels, 1))

    def forward(self, x):
        x = self.lift(x)
        for spectral, local in zip(self.spec, self.local):
            x = F.gelu(spectral(x) + local(x))
        return self.proj(x)


class DeepONet(nn.Module):
    def __init__(self, in_channels=10, out_channels=4, rank=48):
        super().__init__()
        self.out_channels, self.rank = out_channels, rank
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, 24, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(24, 48, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, out_channels*rank),
        )
        self.trunk = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, out_channels*rank))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        b, _, h, w = x.shape
        branch = self.branch(x).view(b, self.out_channels, self.rank)
        coords = x[:, 4:6].permute(0, 2, 3, 1)
        trunk = self.trunk(coords).view(b, h, w, self.out_channels, self.rank)
        return (torch.einsum("bor,bhwor->bohw", branch, trunk) / self.rank**0.5 + self.bias[None, :, None, None])


class PointMLP(nn.Module):
    def __init__(self, in_channels=10, out_channels=4, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 1), nn.SiLU(),
            nn.Conv2d(width, width, 1), nn.SiLU(),
            nn.Conv2d(width, width, 1), nn.SiLU(),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_model(name: str) -> nn.Module:
    return {"unet": UNet, "fno": FNO2d, "deeponet": DeepONet, "pinn": PointMLP}[name]()
