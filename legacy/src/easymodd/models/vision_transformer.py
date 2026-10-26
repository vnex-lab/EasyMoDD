"""Configuration contract for Vision Transformer models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionTransformerConfig:
    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    classes: int = 10
    embed_dim: int = 384
    heads: int = 6
    layers: int = 6
    ff_dim: int = 1536

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if min(self.patch_size, self.channels, self.classes, self.embed_dim, self.heads, self.layers, self.ff_dim) < 1:
            raise ValueError("Vision Transformer dimensions must be positive")
        if self.embed_dim % self.heads:
            raise ValueError("embed_dim must be divisible by heads")

    @property
    def patch_count(self) -> int:
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side
