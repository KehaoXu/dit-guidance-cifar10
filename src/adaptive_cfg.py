import math

import torch
from torch import nn


class AdaptiveCFG(nn.Module):
    '''Predict a bounded classifier-free guidance scale from a timestep.'''

    def __init__(self, num_timesteps: int, hidden_size: int = 32,
                 min_scale: float = 1.0, max_scale: float = 12.5,
                 initial_scale: float = 7.5):
        super().__init__()
        if num_timesteps < 2:
            raise ValueError('num_timesteps must be at least 2')
        if hidden_size <= 0:
            raise ValueError('hidden_size must be positive')
        if min_scale >= max_scale:
            raise ValueError('min_scale must be smaller than max_scale')
        if not min_scale <= initial_scale <= max_scale:
            raise ValueError('initial_scale must lie inside [min_scale, max_scale]')

        self.num_timesteps = num_timesteps
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

        final_layer = self.network[-1]
        nn.init.zeros_(final_layer.weight)
        position = (initial_scale - min_scale) / (max_scale - min_scale)
        position = min(max(position, 1e-6), 1.0 - 1e-6)
        nn.init.constant_(final_layer.bias, math.log(position / (1.0 - position)))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        normalized = timestep.to(dtype=self.network[0].weight.dtype).reshape(-1, 1)
        normalized = normalized / (self.num_timesteps - 1)
        unit_scale = torch.sigmoid(self.network(normalized)).squeeze(-1)
        return self.min_scale + (self.max_scale - self.min_scale) * unit_scale


class IntervalCFG(nn.Module):
    '''Apply a constant guidance scale only inside a noise-level interval.

    Guidance hurts diversity at high noise and amplifies model error at low
    noise, so outside ``[sigma_lo, sigma_hi]`` the scale is 1, i.e. the plain
    conditional model. The interval is defined on sigma = sqrt((1 - abar) / abar)
    rather than on step indices, so it is independent of the sampler's step count.
    '''

    def __init__(self, alphas_cumprod: torch.Tensor, scale: float,
                 sigma_lo: float, sigma_hi: float):
        super().__init__()
        for name, value in (('scale', scale), ('sigma_lo', sigma_lo),
                            ('sigma_hi', sigma_hi)):
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite.')
        if sigma_lo < 0:
            raise ValueError('sigma_lo must be non-negative.')
        if sigma_lo >= sigma_hi:
            raise ValueError('sigma_lo must be smaller than sigma_hi.')

        self.scale = scale
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi

        sigma = ((1.0 - alphas_cumprod) / alphas_cumprod).sqrt()
        inside = (sigma >= sigma_lo) & (sigma <= sigma_hi)
        table = torch.where(inside, torch.full_like(sigma, scale),
                            torch.ones_like(sigma))
        self.register_buffer('table', table.float())

    @property
    def num_active_timesteps(self) -> int:
        return int((self.table != 1.0).sum())

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.table[timestep]
