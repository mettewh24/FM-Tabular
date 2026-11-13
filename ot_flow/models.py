"""Velocity-field models used by the OT flow-matching generator."""

import math
from typing import Optional

import torch


class SinusoidalTimeEmbedding(torch.nn.Module):
    """Sinusoidal positional embedding for scalar time in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMLayer(torch.nn.Module):
    """Feature-wise Linear Modulation conditioned on an embedding."""

    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.scale = torch.nn.Linear(condition_dim, hidden_dim)
        self.shift = torch.nn.Linear(condition_dim, hidden_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        scale = self.scale(condition)
        shift = self.shift(condition)

        if x_norm.ndim > 2:
            expand_shape = (scale.shape[0],) + (1,) * (x_norm.ndim - 2) + (scale.shape[-1],)
            scale = scale.view(*expand_shape)
            shift = shift.view(*expand_shape)

        return x_norm * scale + shift


class TabularFlowNet(torch.nn.Module):
    """MLP velocity field for flow matching on tabular data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        n_classes: int = 0,
        class_embed_dim: int = 64,
    ):
        super().__init__()

        self.n_classes = int(n_classes)
        self.class_embed_dim = int(class_embed_dim) if n_classes > 0 else 0
        condition_dim = time_dim + self.class_embed_dim

        self.time_embed = torch.nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            torch.nn.Linear(time_dim, time_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_dim, time_dim),
        )

        self.class_embed: Optional[torch.nn.Embedding] = (
            torch.nn.Embedding(self.n_classes, self.class_embed_dim)
            if self.n_classes > 0
            else None
        )

        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
            )
            for _ in range(depth)
        ])
        self.film_layers = torch.nn.ModuleList([
            FiLMLayer(hidden_dim, condition_dim) for _ in range(depth)
        ])
        self.out = torch.nn.Linear(hidden_dim, input_dim)

        torch.nn.init.zeros_(self.out.bias)
        torch.nn.init.normal_(self.out.weight, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)

        if self.class_embed is not None and y is not None:
            y_emb = self.class_embed(y.view(-1).long())
            condition = torch.cat([t_emb, y_emb], dim=-1)
        else:
            condition = t_emb

        hidden = self.input_proj(x)
        for layer, film in zip(self.layers, self.film_layers):
            hidden = hidden + film(layer(hidden), condition)

        return self.out(hidden)


__all__ = ["TabularFlowNet"]
