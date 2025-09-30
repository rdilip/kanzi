"""Diffusion autoencoder model"""

import torch
from torch import nn, Tensor
from torch.nn.attention.flex_attention import create_block_mask
import math
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
from math import prod
from jaxtyping import Float, Array
from typing import Optional
from timm.layers import Mlp
from torchdiffeq import odeint
from pathlib import Path
from scipy.spatial.transform import Rotation
import functools


from .attention import TransformerStack, SelfAttention, TransformerBlock
from .fsq import FSQ
from .cfm import ConditionalFlowMatcher


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


def sample_uniform_rotation(
    shape=tuple(), dtype=None, device=None
) -> Float[Array, "*batch 3 3"]:
    """
    Samples rotations distributed uniformly.

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


##################################################################
#                               GPT                              #
##################################################################
@dataclass
class GPTConfig:
    vocab_size: int = 21
    n_channels: int = 128
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0
    block_size: int = 1024
    mlp_factor: int = 4
    bos: int = None
    eos: int = None


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_channels)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_channels=cfg.n_channels,
                    n_heads=cfg.n_heads,
                    mlp_factor=cfg.mlp_factor,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.proj = nn.Linear(cfg.n_channels, cfg.vocab_size)
        self.embed.weight = self.proj.weight
        self.ln = nn.LayerNorm(cfg.n_channels)

        device = self.embed.weight.device

        self.block_mask = create_block_mask(
            self.causal,
            B=None,
            H=None,
            Q_LEN=cfg.block_size,
            KV_LEN=cfg.block_size,
            device=device,
        )

    def to(self, device):
        super().to(device)
        self.block_mask = self.block_mask.to(device)
        return self

    @functools.lru_cache(maxsize=256)
    def get_block_mask(self, L, device):
        return create_block_mask(
            self.causal,
            B=None,
            H=None,
            Q_LEN=L,
            KV_LEN=L,
            device=device,
        )

    @staticmethod
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(cls, ckpt_pth):
        assert isinstance(ckpt_pth, (str, Path)), "ckpt_pth must be a string or Path"
        ckpt = torch.load(ckpt_pth)
        cfg = GPTConfig(**ckpt["model_cfg"])
        # some monkey patching real quick, since we set this manually earlier
        if cfg.bos > cfg.vocab_size - 1:
            cfg.bos = cfg.vocab_size - 2
            cfg.eos = cfg.vocab_size - 1
        model = cls(cfg)
        model.load_state_dict(ckpt["model"])
        return model

    def forward(self, tok_BL, tgt_BL=None):
        s_BLD = self.embed(tok_BL)
        # make mask
        # quick note: we can use the same mask every time, but we should eventually mask out by protein
        # index so we'll need to recompute the mask anyway then.

        device = s_BLD.device
        L = s_BLD.size(-2)

        block_mask = self.get_block_mask(L, device)

        for block in self.blocks:
            s_BLD = block(s_BLD, block_mask, pair_bias_BLLD=None)
        s_BLD = self.ln(s_BLD)

        loss = None
        if tgt_BL is not None:
            logits_BLV = self.proj(s_BLD)
            loss = F.cross_entropy(
                logits_BLV.view(-1, self.cfg.vocab_size),
                tgt_BL.view(-1),
                reduction="none",
            ).mean()
        else:
            logits_BLV = self.proj(s_BLD[:, [-1], :])
        return logits_BLV, loss

    @torch.no_grad()
    def minp_filter(self, logits, p_base=0.1):
        probs = logits.softmax(dim=-1)
        max_token = logits.argmax(dim=-1)
        p_scaled = probs.max(dim=-1).values * p_base
        prob_new = torch.where(
            probs > p_scaled.unsqueeze(-1), probs, torch.zeros_like(probs)
        )
        prob_new = prob_new / prob_new.sum(dim=-1, keepdim=True)

        return prob_new

    @torch.no_grad()
    def nucleus_filter(self, logits, p=0.9):
        """Set logits of tokens outside top-p to -inf."""
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        # Set filtered logits to -inf
        sorted_logits = logits.gather(-1, sorted_indices)
        sorted_logits[mask] = float("-inf")

        # Restore original order
        filtered_logits = torch.empty_like(logits).scatter(
            -1, sorted_indices, sorted_logits
        )
        return filtered_logits

    @torch.no_grad()
    def generate(
        self,
        *,
        max_output_size: int,
        temperature: float = 1.0,
        sampling_method: str = "nucleus",
        threshold: float = 0.9,
    ):
        print(f"Sampling using {sampling_method} with threshold {threshold}")
        device = self.embed.weight.device
        idx = torch.tensor([self.cfg.bos], device=device).unsqueeze(0)

        while idx.size(-1) < max_output_size:
            logits, _ = self(idx)
            logits = logits / temperature
            assert threshold

            logits = logits / temperature
            # logits[..., start_token] = -1.0e8
            if sampling_method == "minp":
                assert threshold < 0.8
                probs = self.minp_filter(logits, p_base=threshold)
            elif sampling_method == "nucleus":
                filtered_logits = self.nucleus_filter(logits, p=threshold)
                probs = F.softmax(filtered_logits, dim=-1)
            else:
                raise ValueError(f"Invalid sampling method: {sampling_method}")

            idx_next = torch.multinomial(probs[:, 0, :], num_samples=1)
            if idx_next.item() == self.cfg.eos:
                break
            idx = torch.cat((idx, idx_next), dim=-1)
            if idx.size(-1) >= max_output_size:
                break
        return idx


##################################################################
#                               DAE                              #
##################################################################


@dataclass
class DAEConfig:
    n_channels_decoder: int
    n_channels_encoder: int
    n_layers_encoder: int
    n_layers_decoder: int
    n_heads: int
    mlp_factor: int
    use_qknorm: bool
    sigma: float = 0.0
    levels: tuple[int] = (8, 8, 8, 8)  # 4096
    drop_cond_p: float = 0.0
    conditioning_type: str = "cat"
    n_channels_pair: int = -1
    n_neighbors: int = 32
    encoder_type: str = "xformer"
    gpt_prior: bool = False
    window_size: int = 10_000  # keep large by default
    share_adaLN: bool = False  # by default False for backwards compatibility, but significantly improves flow matching results


class DAE(nn.Module):
    """DAE has encoder, decoder, VQ, and diffusion loss."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        levels = list(cfg.levels)

        self.quantize = FSQ(
            levels=levels,
            dim_out=cfg.n_channels_decoder,
            dim=cfg.n_channels_encoder,
            jitter_spread=0.0,
        )
        self.codebook_size = prod(levels)

        self.cfm = ConditionalFlowMatcher(cfg.sigma, "uniform_beta")

        # Whether to add an additional per sample random rotation
        # in front of
        self.drop_cond_p = cfg.drop_cond_p
        self.pair_bias = cfg.n_channels_pair > 0

        if self.pair_bias:
            self.pair_embedder = PairEmbedderNoCond(cfg.n_channels_pair, 100)

        assert cfg.encoder_type in [
            "protein-mpnn",
            "rohit-mpnn",
            "xformer",
            "xformer-local",
        ]

        self.up = nn.Sequential(
            nn.Linear(3, cfg.n_channels_encoder),
            nn.SiLU(),
            nn.Linear(cfg.n_channels_encoder, cfg.n_channels_encoder),
            nn.LayerNorm(cfg.n_channels_encoder),
        )
        self.encoder = TransformerStack(
            n_channels=cfg.n_channels_encoder,
            n_heads=cfg.n_heads,
            mlp_factor=cfg.mlp_factor,
            window_size=cfg.window_size,  # hack with no windowing, full bidirectional, but keeping this in case we want to add sliding window attention
            attn_backend="spda",
            n_layers=cfg.n_layers_encoder,
            n_channels_pair=cfg.n_channels_pair,
            is_causal=False,
        )

        self.net = DiT(
            n_channels=cfg.n_channels_decoder,
            n_channels_pair=cfg.n_channels_pair,
            channels_in=3,
            n_layers=cfg.n_layers_decoder,
            n_heads=cfg.n_heads,
            mlp_factor=cfg.mlp_factor,
            use_qknorm=cfg.use_qknorm,
            conditioning_type=cfg.conditioning_type,
            share_adaLN=cfg.share_adaLN,
        )

        if cfg.gpt_prior:
            cfg = GPTConfig(
                vocab_size=prod(cfg.levels),
                n_channels=256,
                n_layers=3,
                n_heads=8,
                dropout=0.1,
                block_size=256,
                mlp_factor=4,
                bos=1001,
                eos=1002,
            )
            self.gpt = GPT(cfg)

    @classmethod
    def from_pretrained(cls, ckpt_pth):
        assert isinstance(ckpt_pth, (str, Path)), "ckpt_pth must be a string or Path"
        ckpt = torch.load(ckpt_pth)
        cfg = DAEConfig(**ckpt["model_cfg"])
        model = cls(cfg)
        model.load_state_dict(ckpt["model"])
        return model

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, x_BLD, preprocess=False):
        # assumes all the same.
        if preprocess:
            x_BLD = x_BLD - x_BLD.mean(dim=1, keepdim=True)
            x_BLD /= 10.
        B, L, D = x_BLD.shape
        # sample a single random orientation to learn so(3) invariant rep
        x_BLD = x_BLD - x_BLD.mean(dim=1, keepdim=True)

        pair_BLLD = None
        if self.pair_bias:
            pair_BLLD = self.pair_embedder(x_BLD)
        s_BLD = self.up(x_BLD)
        s_BLD = self.encoder(s_BLD, pair_bias_BLLD=pair_BLLD)
        c_BLD, idx_BL = self.quantize(s_BLD)

        return s_BLD, c_BLD, idx_BL

    def decode(
        self,
        idx_BL,
        n_steps=100,
        noise_weight=0.45,
        score_weight=1.0,
        cfg_weight=1.0,
        g_fn=None,
    ):
        c_BLD = self.quantize.indices_to_codes(idx_BL)
        device = c_BLD.device
        x_BLD = torch.randn(*c_BLD.shape[:-1], 3, device=device)
        x_BLD = x_BLD - x_BLD.mean(dim=1, keepdim=True)
        t = torch.linspace(0, 1, n_steps, device=device)
        gt = self.get_gt(t)
        dt = t[1] - t[0]

        if isinstance(noise_weight, (list, torch.Tensor, np.ndarray)):
            B = len(noise_weight)
            x_BLD = x_BLD.repeat(B, 1, 1)
            score_weight = torch.tensor(
                score_weight, device=device, dtype=torch.float32
            ).view(-1, 1, 1)
            noise_weight = torch.tensor(
                noise_weight, device=device, dtype=torch.float32
            ).view(-1, 1, 1)
            cfg_weight = torch.tensor(
                cfg_weight, device=device, dtype=torch.float32
            ).view(-1, 1, 1)
            c_BLD = c_BLD.repeat(B, 1, 1)

        for step in range(n_steps):
            t_ = t[step]
            v = self.net(x_BLD, torch.tensor([t_]).to(x_BLD.device), z_BLD=c_BLD)
            v -= v.mean(dim=1, keepdim=True)

            v0 = self.net(
                x_BLD,
                torch.tensor([t_]).to(x_BLD.device),
                z_BLD=torch.zeros_like(c_BLD),
            )
            v0 -= v0.mean(dim=1, keepdim=True)

            vcfg = v0 + cfg_weight * (v - v0)

            if t_ >= 0.99:
                x_BLD = x_BLD + vcfg * dt
            else:
                score = self.vf_to_score(
                    x_BLD, v, t_ * torch.ones(x_BLD.shape[:-1], device=x_BLD.device)
                )  # get score from v, [*, dim]
                score0 = self.vf_to_score(
                    x_BLD, v0, t_ * torch.ones(x_BLD.shape[:-1], device=x_BLD.device)
                )  #

                scorecfg = score0 + cfg_weight * (score - score0)

                eps = torch.randn_like(x_BLD)
                eps -= eps.mean(dim=1, keepdim=True)
                std_eps = torch.sqrt(2 * gt[step] * noise_weight * dt)

                delta_x = (
                    vcfg + gt[step] * scorecfg * score_weight
                ) * dt + std_eps * eps
                x_BLD = x_BLD + delta_x
        return x_BLD

    def forward(self, x_BLD):
        # center data
        x_BLD = x_BLD - x_BLD.mean(dim=1, keepdim=True)
        _, c_BLD, idx_BL = self.encode(x_BLD)

        B, L, D = c_BLD.shape

        x0 = torch.randn_like(x_BLD)
        x0 = x0 - x0.mean(dim=1, keepdim=True)
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(x0, x_BLD)

        # fake classifier free guidance
        cmask = (torch.rand((B,), device=x_BLD.device) > self.drop_cond_p)[
            :, None, None
        ]
        c_BLD = c_BLD * cmask

        vt = self.net(xt, t, z_BLD=c_BLD)
        ut = ut[:, :L, :]
        vt = vt[:, :L, :]

        # align for rotational invariant flow loss
        loss = ((ut[:, :L, :] - vt[:, :L, :]) ** 2).mean()

        # compute gpt prior loss
        loss_gpt = torch.tensor(0.0, device=x_BLD.device)
        idx_BL = idx_BL.contiguous().long()
        if self.cfg.gpt_prior:
            _, loss_gpt = self.gpt(
                idx_BL[:, :-1].contiguous(), idx_BL[:, 1:].contiguous()
            )
            loss_gpt = loss_gpt.mean()

        loss_dict = {"flow_loss": loss, "gpt_prior_loss": loss_gpt}

        return idx_BL, loss_dict

    def get_gt(
        self,
        t: Float[Tensor, "s"],
        mode: str = "us",
        param: float = 1.0,
        clamp_val: Optional[float] = None,
        eps: float = 1e-2,
    ) -> Float[Tensor, "s"]:
        """
        Computes gt for different modes.

        Args:
            t: times where we'll evaluate, covers [0, 1), shape [nsteps]
            mode: "us" or "tan"
            param: parameterized transformation
            clamp_val: value to clamp gt, no clamping if None
            eps: small value leave as it is

        Returns
        """

        # Function to get variants for some gt mode
        def transform_gt(gt, f_pow=1.0):
            # 1.0 means no transformation
            if f_pow == 1.0:
                return gt

            # First we somewhat normalize between 0 and 1
            log_gt = torch.log(gt)
            mean_log_gt = torch.mean(log_gt)
            log_gt_centered = log_gt - mean_log_gt
            normalized = torch.nn.functional.sigmoid(log_gt_centered)
            # Transformation here
            normalized = normalized**f_pow
            # Undo normalization with the transformed variable
            log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
            log_gt_rec = log_gt_centered_rec + mean_log_gt
            gt_rec = torch.exp(log_gt_rec)
            return gt_rec

        # Numerical reasons for some schedule
        t = torch.clamp(t, 0, 1 - 1e-5)

        if mode == "us":
            num = 1.0 - t
            den = t
            gt = num / (den + eps)
        elif mode == "tan":
            num = torch.sin((1.0 - t) * torch.pi / 2.0)
            den = torch.cos((1.0 - t) * torch.pi / 2.0)
            gt = (torch.pi / 2.0) * num / (den + eps)
        elif mode == "1/t":
            num = 1.0
            den = t
            gt = num / (den + eps)
        else:
            raise NotImplementedError(f"gt not implemented {mode}")
        gt = transform_gt(gt, f_pow=param)
        gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
        return gt  # [s]

    def vf_to_score(
        self,
        x_t: Float[Tensor, "* n 3"],
        v: Float[Tensor, "* n 3"],
        t: Float[Tensor, "* n"],
        scale_ref: float = 1.0,
    ):
        """
        Compute score of noisy density given the vector field learned by flow matching. With
        our interpolation scheme these are related by

        v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

        or equivalently,

        s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

        Args:
            x_t: Noisy sample, shape [*, dim]
            v: Vector field, shape [*, dim]
            t: Interpolation time, shape [*]

        Returns:
            Score of intermediate density, shape [*, dim].
        """
        assert torch.all(t < 1.0), "vf_to_score requires t < 1 (strict)"
        num = t[..., None] * v - x_t  # [*, n, 3]
        den = (1.0 - t)[..., None] * scale_ref**2  # [*, n, 1]
        score = num / den
        return score  # [*, dim]


@dataclass
class RnFlowMatcherConfig:
    n_channels: int = 256
    n_channels_pair: int = 64
    channels_in: int = 3
    n_layers: int = 16
    n_heads: int = 8
    mlp_factor: int = 4
    sigma: float = 0.0
    use_qknorm: bool = False
    conditioning_type: Optional[str] = None
    share_adaLN: bool = False


class RnFlowMatcher(nn.Module):
    def __init__(self, cfg: RnFlowMatcherConfig):
        super().__init__()
        self.cfg = cfg
        self.net = DiT(
            n_channels=cfg.n_channels,
            channels_in=cfg.channels_in,
            n_channels_pair=cfg.n_channels_pair,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            mlp_factor=cfg.mlp_factor,
            use_qknorm=cfg.use_qknorm,
            conditioning_type=cfg.conditioning_type,
            share_adaLN=cfg.share_adaLN,
        )
        self.cfm = ConditionalFlowMatcher(cfg.sigma, "uniform_beta")

    def forward(self, x_BLD, z_BLD=None):
        """
        x1 is clean data
        """
        x0 = torch.randn_like(x_BLD)
        x0 = x0 - x0.mean(dim=1, keepdim=True)
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(x0, x_BLD)
        vt = self.net(xt, t)
        loss = (ut - vt) ** 2
        loss_dict = {"flow_loss": loss.mean()}
        return loss_dict

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def sample(self, x_BLD, n_steps):
        device = x_BLD.device

        def _ode_func(t, x):
            x = x - x.mean(dim=1, keepdim=True)
            t = torch.ones(x.size(0)).to(x.device) * t
            return self.net(x, t)

        x = x_BLD
        x = x - x.mean(dim=1, keepdim=True)
        t = torch.linspace(0, 1, n_steps, device=device)
        samples = odeint(_ode_func, x, t)
        return samples

    @torch.no_grad()
    def euler_sample(self, x_BLD, n_steps):
        device = x_BLD.device
        x = x_BLD
        x = x - x.mean(dim=1, keepdim=True)
        t = torch.linspace(0, 1, n_steps, device=device)
        dt = t[1] - t[0]
        for t_ in t[:-1]:
            x = x + self.net(x, torch.tensor([t_]).to(x.device)) * dt
            x -= x.mean(dim=1, keepdim=True)
            # print(x.mean(dim=1).abs().max(), t_)
        return x

    @torch.no_grad()
    def euler_maruyama_sample(
        self, x_BLD, n_steps, score_weight=1.0, noise_weight=0.45
    ):
        device = x_BLD.device
        x = x_BLD
        x = x - x.mean(dim=1, keepdim=True)
        t = torch.linspace(0, 1, n_steps + 1, device=device)[:-1]
        gt = self.get_gt(t)
        dt = t[1] - t[0]
        for step in range(n_steps):
            t_ = t[step]
            v = self.net(x, torch.tensor([t_]).to(x.device))
            v -= v.mean(dim=1, keepdim=True)

            if t_ >= 0.99:
                x = x + v * dt
            else:
                score = self.vf_to_score(
                    x, v, t_ * torch.ones(x.shape[:-1], device=x.device)
                )  # get score from v, [*, dim]
                eps = torch.randn_like(x)
                eps -= eps.mean(dim=1, keepdim=True)
                std_eps = torch.sqrt(2 * gt[step] * noise_weight * dt)
                delta_x = (v + gt[step] * score * score_weight) * dt + std_eps * eps
                x = x + delta_x

            # x -= x.mean(dim=1, keepdim=True)

        return x

    def vf_to_score(
        self,
        x_t: Float[Tensor, "* n 3"],
        v: Float[Tensor, "* n 3"],
        t: Float[Tensor, "* n"],
        scale_ref: float = 1.0,
    ):
        """
        Compute score of noisy density given the vector field learned by flow matching. With
        our interpolation scheme these are related by

        v(x_t, t) = (1 / t) (x_t + scale_ref ** 2 * (1 - t) * s(x_t, t)),

        or equivalently,

        s(x_t, t) = (t * v(x_t, t) - x_t) / (scale_ref ** 2 * (1 - t)).

        Args:
            x_t: Noisy sample, shape [*, dim]
            v: Vector field, shape [*, dim]
            t: Interpolation time, shape [*]

        Returns:
            Score of intermediate density, shape [*, dim].
        """
        assert torch.all(t < 1.0), "vf_to_score requires t < 1 (strict)"
        num = t[..., None] * v - x_t  # [*, n, 3]
        den = (1.0 - t)[..., None] * scale_ref**2  # [*, n, 1]
        score = num / den
        return score  # [*, dim]

    def get_gt(
        self,
        t: Float[Tensor, "s"],
        mode: str = "us",
        param: float = 1.0,
        clamp_val: Optional[float] = None,
        eps: float = 1e-2,
    ) -> Float[Tensor, "s"]:
        """
        Computes gt for different modes.

        Args:
            t: times where we'll evaluate, covers [0, 1), shape [nsteps]
            mode: "us" or "tan"
            param: parameterized transformation
            clamp_val: value to clamp gt, no clamping if None
            eps: small value leave as it is

        Returns
        """

        # Function to get variants for some gt mode
        def transform_gt(gt, f_pow=1.0):
            # 1.0 means no transformation
            if f_pow == 1.0:
                return gt

            # First we somewhat normalize between 0 and 1
            log_gt = torch.log(gt)
            mean_log_gt = torch.mean(log_gt)
            log_gt_centered = log_gt - mean_log_gt
            normalized = torch.nn.functional.sigmoid(log_gt_centered)
            # Transformation here
            normalized = normalized**f_pow
            # Undo normalization with the transformed variable
            log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
            log_gt_rec = log_gt_centered_rec + mean_log_gt
            gt_rec = torch.exp(log_gt_rec)
            return gt_rec

        # Numerical reasons for some schedule
        t = torch.clamp(t, 0, 1 - 1e-5)

        if mode == "us":
            num = 1.0 - t
            den = t
            gt = num / (den + eps)
        elif mode == "tan":
            num = torch.sin((1.0 - t) * torch.pi / 2.0)
            den = torch.cos((1.0 - t) * torch.pi / 2.0)
            gt = (torch.pi / 2.0) * num / (den + eps)
        elif mode == "1/t":
            num = 1.0
            den = t
            gt = num / (den + eps)
        else:
            raise NotImplementedError(f"gt not implemented {mode}")
        gt = transform_gt(gt, f_pow=param)
        gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
        return gt  # [s]


##################################################################################
#                             Components for RnFlowMatcher                       #
##################################################################################


class PairEmbedderNoCond(nn.Module):
    def __init__(self, n_channels, n_buckets):
        """
        Pair embedding -- this assumes that ALL elements in a batch are the same, just rotated. Thus, the inter pair
        distances won't change. This will save a ton of memory and we've proven we can train this way anyway.
        """
        super().__init__()
        self.embed = nn.Embedding(n_buckets, n_channels)
        self.pos_embed = nn.Embedding(128, n_channels)
        # 30 Angstoms = 3 nm, inputs are in nm
        # we compute bins on the square, avoids a sqrt which can get nasty with gradients
        self.register_buffer("bins", torch.linspace(0, 3**2, n_buckets - 1))
        self.mlp = Mlp(
            n_channels, 4 * n_channels, n_channels, act_layer=nn.GELU, norm_layer=None
        )
        self.ln = nn.LayerNorm(n_channels)
        self.nc = n_channels

    def forward(self, x_BLD):
        B, L, D = x_BLD.shape
        # just use the first element, assumes that these are repeated
        # if self.training:
        #    d_BLL = ((x_BLD[0, :, None] - x_BLD[0, None, :]) ** 2).sum(dim=-1)
        #    tmp = ((x_BLD[1, :, None] - x_BLD[1, None, :]) ** 2).sum(dim=-1)
        #    assert (d_BLL - tmp).abs().max() < 1e-2
        #    d_BLL = d_BLL.unsqueeze(0).expand(B, -1, -1).contiguous()
        # else:
        d_BLL = ((x_BLD[:, :, None] - x_BLD[:, None, :]) ** 2).sum(dim=-1)

        idx_L = torch.arange(L, device=x_BLD.device)
        idx_LL = (idx_L[:, None] - idx_L[None, :]).clip(min=-64, max=63) + 64
        pos_LLD = self.pos_embed(idx_LL)

        d_BLL = torch.bucketize(d_BLL, self.bins)
        s_BLLD = self.embed(d_BLL) + pos_LLD
        s_BLLD = self.mlp(self.ln(s_BLLD))

        return s_BLLD


class PairEmbedder(nn.Module):
    def __init__(self, n_channels, n_buckets):
        """
        Pair embedding -- this assumes that ALL elements in a batch are the same, just rotated. Thus, the inter pair
        distances won't change. This will save a ton of memory and we've proven we can train this way anyway.
        """
        super().__init__()
        self.embed = nn.Embedding(n_buckets, n_channels)
        self.pos_embed = nn.Embedding(128, n_channels)
        # 30 Angstoms = 5 nm, inputs are in nm
        self.register_buffer("bins", torch.linspace(0, 3, n_buckets - 1))
        self.mlp = Mlp(
            n_channels, 4 * n_channels, n_channels, act_layer=nn.GELU, norm_layer=None
        )
        self.ln = nn.LayerNorm(n_channels)
        self.nc = n_channels

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_channels, n_channels * 3, bias=True),
        )

    @torch.compile()
    def forward(self, x_BLD, c_BD):
        B, L, D = x_BLD.shape
        # just use the first element, assumes that these are repeated
        if self.training:
            d_BLL = (x_BLD[0, :, None] - x_BLD[0, None, :]) ** 2
            tmp = d_BLL = (x_BLD[1, :, None] - x_BLD[1, None, :]) ** 2
            assert d_BLL.allclose(tmp)
            d_BLL = d_BLL.unsqueeze(0).expand(B, -1, -1)
        else:
            d_BLL = ((x_BLD[:, :, None] - x_BLD[:, None, :]) ** 2).sum(dim=-1)

        idx_L = torch.arange(L, device=x_BLD.device)
        idx_LL = (idx_L[:, None] - idx_L[None, :]).clip(min=-64, max=63) + 64
        pos_LLD = self.pos_embed(idx_LL)

        shift_time, scale_time, gate_time = self.adaLN_modulation(c_BD).chunk(3, dim=-1)
        shift_time = shift_time.view(B, 1, 1, self.nc)
        scale_time = scale_time.view(B, 1, 1, self.nc)
        gate_time = gate_time.view(B, 1, 1, self.nc)

        d_BLL = torch.bucketize(d_BLL, self.bins)
        s_BLLD = self.embed(d_BLL) + pos_LLD
        s_BLLD = s_BLLD * (1 + scale_time) + shift_time

        s_BLLD = s_BLLD + gate_time * self.mlp(self.ln(s_BLLD))

        return s_BLLD


class DiTBlock(nn.Module):
    def __init__(
        self,
        *,
        n_channels,
        n_channels_pair,
        n_heads,
        mlp_factor,
        use_qknorm=False,
        dropout=0.1,
        shared_adaLN=None,
        attn_backend=True,
    ):
        super().__init__()
        self.attn_backend = "spda"
        self.attn = SelfAttention(
            n_channels,
            n_heads,
            backend=self.attn_backend,
            dropout=dropout,
            use_qknorm=use_qknorm,
        )
        self.mlp = Mlp(
            n_channels,
            n_channels * mlp_factor,
            n_channels,
            act_layer=nn.GELU,
            norm_layer=None,
            drop=dropout,
        )
        self.norm1 = nn.LayerNorm(n_channels, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(n_channels, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(n_channels, elementwise_affine=False)

        if shared_adaLN is not None:
            self.adaLN_modulation = shared_adaLN
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_channels, n_channels * 6, bias=True),
            )

    def _get_attn_kwargs(self):
        if self.attn_backend == "spda":
            attn_kwargs = {"attn_mask": None}
        elif self.attn_backend == "flex":
            attn_kwargs = {"block_mask": None}
        else:
            attn_kwargs = {}
        return attn_kwargs

    def forward(self, s_BLD, c_BD, z_BLD=None):
        attn_kwargs = self._get_attn_kwargs()
        adaLN_output = self.adaLN_modulation(c_BD)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            adaLN_output.chunk(6, dim=-1)
        )
        s_BLD = s_BLD + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(s_BLD), shift_msa, scale_msa), **attn_kwargs
        )

        s_BLD = s_BLD + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(s_BLD), shift_mlp, scale_mlp)
        )
        return s_BLD


class DiT(nn.Module):
    def __init__(
        self,
        *,
        n_channels,
        n_channels_pair,
        channels_in,
        n_layers,
        n_heads,
        conditioning_type="cat",
        mlp_factor=4,
        use_qknorm=False,
        share_adaLN=True,
    ):
        super().__init__()

        self.up = InitialLinear(channels_in, n_channels)
        self.share_adaLN = share_adaLN

        if share_adaLN:
            self.shared_adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_channels, n_channels * 6, bias=True),
            )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    n_channels=n_channels,
                    n_channels_pair=n_channels_pair,
                    n_heads=n_heads,
                    mlp_factor=mlp_factor,
                    use_qknorm=use_qknorm,
                    shared_adaLN=self.shared_adaLN_modulation if share_adaLN else None,
                )
                for _ in range(n_layers)
            ]
        )
        self.time_embedder = TimestepEmbedder(n_channels)

        # I don't like modules that conditionally change the state dict
        self.conditioning_type = conditioning_type
        assert conditioning_type in ["cat", "xatt", None]

        if self.conditioning_type == "cat":
            self.cond_embed = nn.Embedding(2, n_channels)
            # self.latent_proj = nn.Sequential(
            #     nn.LayerNorm(n_channels, elementwise_affine=False),
            #     nn.Linear(n_channels, n_channels, bias=True),
            #     nn.SiLU(),
            #     nn.Linear(n_channels, n_channels, bias=True),
            # )
        elif self.conditioning_type == "xatt":
            pass
        else:
            raise ValueError(f"Invalid conditioning type: {self.conditioning_type}")

        self.final = FinalLinear(n_channels, channels_in)

    def forward(self, x_BLD, t_B, z_BLD=None):
        """
        z_BLD is concatenated to the input as conditioning vector.
        """
        c_BD = self.time_embedder(t_B)
        s_BLD = self.up(x_BLD, c_BD)

        # c_BD = t_BD
        if self.conditioning_type == "cat":
            # in context conditioning
            assert z_BLD is not None, (
                "z_BLD must be provided if has_conditioning is True"
            )
            shape, dev = z_BLD.shape[:-1], z_BLD.device
            cond_BL = torch.cat(
                (
                    torch.zeros(shape, dtype=torch.long, device=dev),
                    torch.ones(shape, dtype=torch.long, device=dev),
                ),
                dim=-1,
            )
            s_BLD = torch.cat([s_BLD, z_BLD], dim=-2)
            s_BLD = s_BLD + self.cond_embed(cond_BL)
            z_BLD = None

            # 1 strong path
            # removing this for now, let's see if it makes adifference
            # z_pool_BD = z_BLD.mean(dim=1)
            # c_BD = c_BD + self.latent_proj(z_pool_BD)

        for block in self.blocks:
            s_BLD = block(s_BLD, c_BD, z_BLD=z_BLD)

        if self.conditioning_type == "cat":
            L = x_BLD.size(1)
            s_BLD = s_BLD[:, :L, :]

        xout_BLD = self.final(s_BLD, c_BD)
        return xout_BLD


class InitialLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.norm_initial = nn.LayerNorm(
            out_channels, elementwise_affine=False, eps=1e-6
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(out_channels, 2 * out_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.linear(x)
        x = modulate(self.norm_initial(x), shift, scale)
        return x


class FinalLinear(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
