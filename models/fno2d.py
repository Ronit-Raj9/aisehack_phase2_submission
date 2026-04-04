"""2D Fourier neural operator baseline (multi-channel weather + emissions → PM2.5 horizons).

New architectures can live as additional modules under ``models/`` and be exported from ``models/__init__.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, n_layers=2):
        super().__init__()

        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        layers = []
        for i in range(n_layers):
            if i == 0:
                inp = in_channels
                out = hidden_channels if n_layers > 1 else out_channels
            elif i == n_layers - 1:
                inp = hidden_channels
                out = out_channels
            else:
                inp = hidden_channels
                out = hidden_channels

            layers.append(nn.Conv1d(inp, out, 1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)

        x = x.view(B, -1, H, W)
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2

        self.weights = nn.Parameter(
            (1 / (in_channels * out_channels))
            * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        out_ft = torch.zeros(
            B,
            self.weights.shape[1],
            H,
            W // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights,
        )

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()

        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.pointwise = nn.Conv2d(width, width, 1)

        self.mlp = ChannelMLP(
            in_channels=width,
            hidden_channels=int(width * 0.5),
            out_channels=width,
            n_layers=2,
        )

    def forward(self, x):
        x = self.spectral(x) + self.pointwise(x)
        x = F.gelu(x)
        x = x + self.mlp(x)
        return x


class FNO2D(nn.Module):
    def __init__(
        self,
        time_in,
        features,
        time_out,
        width=20,
        modes=12,
    ):
        super().__init__()

        self.time_in = time_in
        self.features = features
        self.time_out = time_out
        self.width = width

        in_channels = time_in * features + 2

        self.lifting = ChannelMLP(
            in_channels=in_channels,
            hidden_channels=width * 2,
            out_channels=width,
            n_layers=2,
        )

        self.block0 = FNOBlock(width, modes)
        self.block1 = FNOBlock(width, modes)
        self.block2 = FNOBlock(width, modes)

        self.proj1 = nn.Conv2d(width, 128, 1)
        self.proj2 = nn.Conv2d(128, time_out, 1)

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)

        gridx = gridx.view(1, 1, nx, 1).repeat(b, 1, 1, ny)
        gridy = gridy.view(1, 1, 1, ny).repeat(b, 1, nx, 1)

        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):

        B, T, nx, ny, f = x.shape

        x = x.permute(0, 2, 3, 1, 4)
        x = x.reshape(B, nx, ny, T * f)
        x = x.permute(0, 3, 1, 2)

        grid = self.get_grid(B, nx, ny, x.device)
        x = torch.cat((x, grid), dim=1)

        x = self.lifting(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)

        x = F.gelu(self.proj1(x))
        x = self.proj2(x)

        return x.permute(0, 2, 3, 1)
