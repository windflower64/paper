"""PartialNet PAT_sf module for the HGNetv2 adapter.

This file is a logic-preserving extraction of the ``channel_type='self'``
branch from ``models/partialnet.py`` in the official PartialNet repository:
https://github.com/haiduo/PartialNet, commit
679cfc0d6f54872686db308de0c4068ee7caf084 (MIT license).

Only unused PartialNet network-building code was omitted.  The channel split,
3x3 partial convolution, LayerNorm2d, four-head self-attention, dropout, and
image relative-position encoding configuration are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from timm.layers import LayerNorm2d

from .partialnet_irpe import build_rpe, get_rpe_config


class RPEAttention(nn.Module):
    """Multi-head self-attention with PartialNet's image RPE."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rpe_config=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(
            rpe_config,
            head_dim=head_dim,
            num_heads=num_heads,
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = x.view(batch, channels, height * width).transpose(1, 2)
        batch, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch,
                tokens,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        if self.rpe_k is not None:
            attn += self.rpe_k(q, height, width)
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v

        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(1, 2).view(batch, channels, height, width)


class PartialSelfAttentionConv(nn.Module):
    """Official PAT_sf split: local Conv3x3 plus global RPE attention."""

    def __init__(self, dim, n_div=4):
        super().__init__()
        dim = int(dim)
        n_div = int(n_div)
        if n_div <= 1 or dim % n_div != 0:
            raise ValueError(
                "PAT_sf requires n_div > 1 and dim divisible by n_div; "
                f"got dim={dim}, n_div={n_div}"
            )

        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        num_heads = 4
        if self.dim_untouched % num_heads != 0:
            raise ValueError(
                "PAT_sf attention channels must be divisible by four heads; "
                f"got {self.dim_untouched}"
            )

        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3,
            self.dim_conv3,
            3,
            1,
            1,
            bias=False,
        )
        rpe_config = get_rpe_config(
            ratio=20,
            method="euc",
            mode="bias",
            shared_head=False,
            skip=0,
            rpe_on="k",
        )
        self.attn = RPEAttention(
            self.dim_untouched,
            num_heads=num_heads,
            attn_drop=0.1,
            proj_drop=0.1,
            rpe_config=rpe_config,
        )
        self.norm = LayerNorm2d(self.dim_untouched)

    @property
    def conv_channels(self):
        return self.dim_conv3

    @property
    def attention_channels(self):
        return self.dim_untouched

    def forward(self, x: Tensor) -> Tensor:
        conv_x, attention_x = torch.split(
            x,
            [self.dim_conv3, self.dim_untouched],
            dim=1,
        )
        conv_x = self.partial_conv3(conv_x)
        attention_x = self.norm(attention_x)
        attention_x = self.attn(attention_x)
        return torch.cat((conv_x, attention_x), dim=1)


class PartialConvOnly(nn.Module):
    """Official PartialNet partial-convolution control (split_cat path)."""

    def __init__(self, dim, n_div=4):
        super().__init__()
        dim = int(dim)
        n_div = int(n_div)
        if n_div <= 1 or dim % n_div != 0:
            raise ValueError(
                "PartialConv requires n_div > 1 and dim divisible by n_div; "
                f"got dim={dim}, n_div={n_div}"
            )
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3,
            self.dim_conv3,
            3,
            1,
            1,
            bias=False,
        )

    @property
    def conv_channels(self):
        return self.dim_conv3

    @property
    def attention_channels(self):
        return 0

    def forward(self, x: Tensor) -> Tensor:
        conv_x, untouched_x = torch.split(
            x,
            [self.dim_conv3, self.dim_untouched],
            dim=1,
        )
        conv_x = self.partial_conv3(conv_x)
        return torch.cat((conv_x, untouched_x), dim=1)


class GlobalQueryAttention(nn.Module):
    """Content-selective global context with a single learned query token.

    Unlike full self-attention, this module forms only one query from the
    spatial mean and attends that query to all keys/values.  The selected
    global vector is then broadcast to every spatial location.  Attention-map
    complexity is therefore O(N), rather than O(N^2).
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        uniform_attention=False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"Global-query channels ({dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.uniform_attention = bool(uniform_attention)
        # Separate projections avoid computing a query for every spatial token.
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        token_count = tokens.shape[1]

        global_token = tokens.mean(dim=1, keepdim=True)
        q = self.q(global_token).reshape(
            batch, 1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = q * self.scale

        kv = self.kv(tokens).reshape(
            batch,
            token_count,
            2,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        if self.uniform_attention:
            # Accuracy control: preserve the same projection topology while
            # removing input-dependent spatial selection.  A deploy version
            # can omit q/k entirely and compute mean(V) directly.
            attn = torch.full_like(attn, 1.0 / token_count)
        attn = self.attn_drop(attn)
        global_context = attn @ v
        global_context = global_context.transpose(1, 2).reshape(
            batch, 1, channels
        )
        global_context = self.proj_drop(self.proj(global_context))
        return global_context.transpose(1, 2).reshape(
            batch, channels, 1, 1
        ).expand(batch, channels, height, width)


class PartialGlobalQueryConv(nn.Module):
    """PAT split with local Conv3x3 and linear-cost global-query context."""

    def __init__(self, dim, n_div=4):
        super().__init__()
        dim = int(dim)
        n_div = int(n_div)
        if n_div <= 1 or dim % n_div != 0:
            raise ValueError(
                "PartialGlobalQueryConv requires n_div > 1 and divisible dim; "
                f"got dim={dim}, n_div={n_div}"
            )
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3,
            self.dim_conv3,
            3,
            1,
            1,
            bias=False,
        )
        self.norm = LayerNorm2d(self.dim_untouched)
        self.attn = GlobalQueryAttention(
            self.dim_untouched,
            num_heads=4,
            attn_drop=0.1,
            proj_drop=0.1,
        )

    @property
    def conv_channels(self):
        return self.dim_conv3

    @property
    def attention_channels(self):
        return self.dim_untouched

    def forward(self, x: Tensor) -> Tensor:
        conv_x, global_x = torch.split(
            x,
            [self.dim_conv3, self.dim_untouched],
            dim=1,
        )
        conv_x = self.partial_conv3(conv_x)
        global_x = self.attn(self.norm(global_x))
        return torch.cat((conv_x, global_x), dim=1)


class PartialGlobalMeanConv(PartialGlobalQueryConv):
    """Strict global-mean control for PartialGlobalQueryConv."""

    def __init__(self, dim, n_div=4):
        super().__init__(dim=dim, n_div=n_div)
        self.attn.uniform_attention = True


__all__ = [
    "GlobalQueryAttention",
    "PartialConvOnly",
    "PartialGlobalMeanConv",
    "PartialGlobalQueryConv",
    "PartialSelfAttentionConv",
    "RPEAttention",
]
