"""Configuration contract for decoder-only text Transformers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextTransformerConfig:
    """Validated architecture settings for a causal text model."""

    vocab_size: int
    embed_dim: int = 512
    heads: int = 8
    layers: int = 4
    ff_dim: int = 2048
    sequence_length: int = 512

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.embed_dim, self.heads, self.layers, self.ff_dim) < 1:
            raise ValueError("Text Transformer dimensions must be positive")
        if self.embed_dim % self.heads:
            raise ValueError("embed_dim must be divisible by heads")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.heads
