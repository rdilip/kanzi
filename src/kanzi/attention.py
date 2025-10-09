import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


from timm.layers import Mlp

from .rotary import RotaryEmbedding

import math


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TransformerStack(nn.Module):
    def __init__(
        self,
        *,
        n_channels,
        n_heads,
        mlp_factor,
        window_size: int,
        n_layers: int,
        attn_backend: str = "flex",
        dropout=0.1,
        n_channels_pair=None,
        is_causal=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_channels=n_channels,
                    n_heads=n_heads,
                    mlp_factor=mlp_factor,
                    attn_backend=attn_backend,
                    dropout=dropout,
                    pair_bias=n_channels_pair > 0,
                    n_channels_pair=n_channels_pair,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = norm
        self.window_size = window_size
        self.is_causal = is_causal

        self.attn_backend = attn_backend
        self._initialize_masking()

    def _initialize_masking(self):
        if self.attn_backend == "flex":

            def sliding_window_causal(b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx if self.is_causal else True
                window_mask = abs(q_idx - kv_idx) <= self.window_size
                return causal_mask & window_mask

            self.window_mask = sliding_window_causal
        else:
            MAX_LEN = 256  # bad bad bad
            idx = torch.arange(MAX_LEN)
            assert not self.is_causal  # nah don't need this
            window_mask = (
                (idx[:, None] - idx[None, :]).abs() <= self.window_size
            ).unsqueeze(0)
            self.register_buffer("window_mask", window_mask, persistent=False)

    def forward(self, s_BLD, pair_bias_BLLD=None):
        # batch is broadcasted
        # get device real quick
        device = next(self.parameters()).device
        if self.attn_backend == "flex":
            attn_kwargs = {
                "block_mask": create_block_mask(
                    self.window_mask,
                    B=None,
                    H=None,
                    Q_LEN=s_BLD.size(-2),
                    KV_LEN=s_BLD.size(-2),
                    device=device,
                )
            }
        else:
            attn_kwargs = {
                "attn_mask": self.window_mask[:, : s_BLD.size(-2), : s_BLD.size(-2)]
            }

        for block in self.blocks:
            s_BLD = block(s_BLD, pair_bias_BLLD=pair_bias_BLLD, **attn_kwargs)
        return s_BLD


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        n_channels,
        n_heads,
        mlp_factor,
        attn_backend: str = "flex",
        dropout=0.1,
        pair_bias=False,
        n_channels_pair=None,
    ):
        super().__init__()

        self.attn_backend = attn_backend
        self.attn = SelfAttention(
            dim=n_channels, n_heads=n_heads, dropout=dropout, backend=attn_backend
        )
        self.mlp = Mlp(
            n_channels,
            n_channels * mlp_factor,
            n_channels,
            act_layer=nn.GELU,
            norm_layer=None,
            drop=dropout,
        )

        self.pair_bias = pair_bias
        if pair_bias:
            self.proj_pair_bias = nn.Linear(n_channels_pair, 1, bias=True)
            self.pair_ln = nn.LayerNorm(n_channels_pair)
        else:
            self.proj_pair_bias = None

    def _get_pair_bias(self, pair_bias_BLLD, **attn_kwargs):
        """Returns pair bias. This can be either a mask OR a score mod depending on the
        attention backend.
        """
        if self.attn_backend == "flex":
            pair_bias_BLL = self.proj_pair_bias(self.pair_ln(pair_bias_BLLD)).squeeze(
                -1
            )

            def pair_biased_attn(score, b, h, q_idx, kv_idx):
                return score + pair_bias_BLL[b, q_idx, kv_idx]

            attn_kwargs["score_mod"] = pair_biased_attn
        elif self.attn_backend == "spda":
            pair_bias_BLL = self.proj_pair_bias(self.pair_ln(pair_bias_BLLD)).squeeze(
                -1
            )
            pair_bias_BLL = torch.where(
                attn_kwargs["attn_mask"], pair_bias_BLL, -torch.inf
            )
            attn_kwargs["attn_mask"] = pair_bias_BLL.unsqueeze(1) # add head dimension
        return attn_kwargs

    def forward(self, s_BLD, pair_bias_BLLD, **attn_kwargs):
        """
        c_BL is time conditioning. eventually the other conditioning is concatenated on.
        """
        score_mod = None
        if self.pair_bias:
            attn_kwargs = self._get_pair_bias(pair_bias_BLLD, **attn_kwargs)

        s_BLD = s_BLD + self.attn(norm(s_BLD), **attn_kwargs)
        s_BLD = s_BLD + self.mlp(norm(s_BLD))
        return s_BLD


class CrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.dim_per_head = dim // n_heads

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=True)

        self.to_out = nn.Linear(dim, dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, s_BLD, c_BLD, block_mask=None, score_mod=None):
        B, L, D = s_BLD.shape
        q = self.to_q(s_BLD)
        k, v = self.to_kv(c_BLD).split(self.dim, dim=-1)

        q = q.view(B, L, self.n_heads, self.dim_per_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.dim_per_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.dim_per_head).transpose(1, 2)

        y = flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)
        y = self.attn_dropout(y)
        y = y.transpose(1, 2).contiguous().view(B, L, self.dim)
        y = self.resid_dropout(self.to_out(y))
        return y


class SelfAttention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, use_qknorm=False, backend="flex", dropout=0.1
    ):
        super().__init__()
        # NOTE: this is a good place to implement pair bias via score mod
        self.dim = dim
        self.n_heads = n_heads
        assert self.dim % self.n_heads == 0
        self.dim_per_head = self.dim // self.n_heads
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.use_qknorm = use_qknorm

        self.pos_embed = RotaryEmbedding(self.dim_per_head)
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.to_out = nn.Linear(dim, dim)
        self.resid_dropout = nn.Dropout(dropout)

        assert backend in ["flex", "spda", "none"]
        self.backend = backend

    def forward(self, x_BLD, **attn_kwargs):  # mask_BLL=None, score_mod=None):
        # incorporate pair bias via score_mod
        B, L, D = x_BLD.shape

        q, k, v = self.to_qkv(x_BLD).split(self.dim, dim=-1)
        q = q.view(B, L, self.n_heads, self.dim_per_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.dim_per_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.dim_per_head).transpose(1, 2)

        if self.use_qknorm:
            q = norm(q)
            k = norm(k)

        q, k = self.pos_embed(q, k)

        if self.backend == "flex":
            y = flex_attention(
                q,
                k,
                v,
                block_mask=attn_kwargs["block_mask"],
                score_mod=attn_kwargs["score_mod"],
            )
        elif self.backend == "spda":
            y = F.scaled_dot_product_attention(q, k, v, **attn_kwargs)
        else:
            # if not pair bias, do attention manually
            raise ValueError("This may not be the best idea")
            sim_BHLL = q @ k.transpose(-2, -1) / math.sqrt(self.dim_per_head)
            sim_BHLL = sim_BHLL + pair_bias_BLL.unsqueeze(-3)
            if self.is_causal:
                mask_LL = torch.ones((L, L), device=sim_BHLL.device, dtype=bool)
                sim_BHLL.masked_fill_(torch.triu(mask_LL, diagonal=1), float("-inf"))

            sim_BHLL = sim_BHLL.softmax(dim=-1)
            y = sim_BHLL @ v

        y = self.attn_dropout(y)
        y = y.transpose(1, 2).contiguous().view(B, L, self.dim)
        y = self.resid_dropout(self.to_out(y))
        return y
