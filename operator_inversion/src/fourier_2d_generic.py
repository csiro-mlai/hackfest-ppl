"""
This file provides the Fourier Neural Operator for times-series-of-2D problems.
"""

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat


def encode_positions(
        dim_sizes,
        pos_low=-1, pos_high=1,
        device=None):
    # dim_sizes is a list of dimensions in all positional/time dimensions
    # e.g. for a 64 x 64 image over 20 steps, dim_sizes = [64, 64, 20]

    # A way to interpret `pos` is that we could append `pos` directly
    # to the raw inputs to attach the positional info to the raw features.
    def generate_grid(size):
        return torch.linspace(
            pos_low, pos_high, steps=size,
            device=device)
    grid_list = list(map(generate_grid, dim_sizes))
    pos = torch.stack(torch.meshgrid(*grid_list), dim=-1)
    # pos.shape == [*dim_sizes, n_dims]
    return pos


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes, residual=True, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.linear = nn.Linear(in_dim, out_dim)
        self.residual = residual
        self.act = nn.ReLU(inplace=True)

        fourier_weight = [nn.Parameter(torch.FloatTensor(
            in_dim, out_dim, n_modes, n_modes, 2)) for _ in range(2)]
        self.fourier_weight = nn.ParameterList(fourier_weight)
        for param in self.fourier_weight:
            nn.init.xavier_normal_(param, gain=1/(in_dim*out_dim))

    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        B, M, N, I = x.shape
        res = self.linear(x)
        # res.shape == [batch_size, grid_size, grid_size, out_dim]

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x_ft = torch.fft.rfft2(x, s=(M, N), norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=4)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft = torch.zeros(B, I, N, M // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :self.n_modes, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, :self.n_modes, :self.n_modes], self.fourier_weight[0])

        out_ft[:, :, -self.n_modes:, :self.n_modes] = self.complex_matmul_2d(
            x_ft[:, :, -self.n_modes:, :self.n_modes], self.fourier_weight[1])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])

        x = torch.fft.irfft2(out_ft, s=(N, M), norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        if self.residual:
            x = self.act(x + res)
        return x


class SimpleBlock2dGeneric(nn.Module):
    def __init__(self,
            modes1, width, input_dim=12,
            dim_sizes = (256,256),
            pos_low=-1.0, pos_high=1.0,
            dropout=0.1, n_layers=4,
            use_position: bool = True,
            max_freq: int = 32,
            num_freq_bands: int = 8,
            freq_base: int = 2,
            n_history: int = 1,
            # state_dim: int = 1,
            # latent_dim: int = 1,
            latent: bool=True,
            # forcing_dim: int = 0,
            forcing: bool=False,
            # param_dim: int = 0,
            param: bool=False,
            n_horizon: int = 1,
            use_phase_position: bool = False,
            residual: bool=False,
            conv_residual: bool=True):
        """
        for non-trivial problems we need to make latent, and forcing and param args be integers rather than flags, and we will always have an extra dimension in the data
        """
        super().__init__()

        self.modes1 = modes1
        self.width = width
        self.residual = residual
        self.dim_sizes = dim_sizes
        self.pos_low = pos_low
        self.pos_high = pos_high
        self.n_history = n_history
        self.latent = latent
        self.forcing = forcing
        self.param = param
        self.n_horizon = n_horizon
        self.register_buffer('_float', torch.FloatTensor([0.1]))

        self.input_dim = (
            n_history + # scalar state
            n_history * forcing +
            latent +
            param +
            len(self.dim_sizes)
        )
        self.in_proj = nn.Linear(self.input_dim, self.width)

        self.spectral_layers = nn.ModuleList([])
        for i in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(
                in_dim=width,
                out_dim=width,
                n_modes=modes1,
                residual=conv_residual,
                dropout=dropout))

        self.out_proj = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1))

    def forward(self, batch):
        # batch is a dict of 'x', 'latent'
        # x.shape == latent.shape == [n_batches, *dim_sizes, input_size]
        x = self._build_features(batch)
        x = self.in_proj(x)
        for layer in self.spectral_layers:
            x = layer(x) + x if self.residual else layer(x)
        x = self.out_proj(x)
        # x.shape == [n_batches, *dim_sizes, 1]
        return {
            'forecast': x
        }

    def _encode_positions(self):
        return encode_positions(
            dim_sizes=self.dim_sizes,
            pos_low=self.pos_low,
            pos_high=self.pos_high,
            device=self._float.device)

    def _build_features(self, batch):
        """
        feature builder for scalar forcing and latent fields
        Takes a dict of state, latent, forcing params and builds a stacked array with position information baked in.
        """
        # We need to repack this so that the in_channels consists of the first n_history time steps
        # and  positional encodings and forcings and the latents and viscosity,
        # where the latents and the positional encodings are shared over time but
        # forcings not (NB in this experiment static forcings count as latents)
        # x.shape == [batch_size, x_dim, y_dim, in_channels]
        # data.shape == [batch_size, *dim_sizes]

        # batch['x'].shape == [batch_size, x_dim, y_dim, n_history]
        # batch['latent'].shape == [batch_size, x_dim, y_dim]

        B, *dim_sizes, T = batch['x'].shape
        m, n = dim_sizes

        x = [batch['x']]
        if self.latent:
            x.append(rearrange(batch['latent'], 'b m n -> b m n 1'))

        pos_feats = self._encode_positions()
        # pos_feats.shape == [*dim_sizes, pos_size]
        pos_feats = repeat(pos_feats, '... -> b ...', b=B)
        # pos_feats.shape == [batch_size, *dim_sizes, n_dims]

        x.append(pos_feats)

        return torch.cat(x, dim=-1)
        # xx.shape == [batch_size, *dim_sizes, self.input_dum]
