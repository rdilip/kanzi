"""
code to train codebook flow model
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from jaxtyping import Float, Array
from pathlib import Path
from functools import partial
import wandb
import math
from dataclasses import dataclass, asdict
import os
from collections import defaultdict
import numpy as np
import socket
import json
import random
import sys
from math import prod

from .utils import sample_uniform_rotation
from .models import DAE, DAEConfig


################################################################################
#                                  utilities                                   #
################################################################################

DATASET_PTH = Path("datasets/memmaps/afdb")
META = json.load(open(DATASET_PTH / "meta.test.json", "r"))
POS_OFFSET = np.prod(META["pos_shape"])
dtype = META["dtype"]
assert dtype == "float32"
dtype = torch.float32 if dtype == "float32" else torch.float16

TRAIN_STARTS = np.load(DATASET_PTH / "starts.train.npy")
TEST_STARTS = np.load(DATASET_PTH / "starts.test.npy")
STARTS = {"train": TRAIN_STARTS, "test": TEST_STARTS}


def load_from_mmap(split, atoms: str, n_repeats=128, unit="nm", device="cuda:0"):
    assert unit in ["nm", "angstrom"]

    pos_mmap = np.memmap(
        DATASET_PTH / f"positions.{split}.dat", dtype=np.float32, mode="r"
    )
    ix = int(random.uniform(0, 1) * (len(STARTS[split]) - 1))
    r, c = STARTS[split][ix], STARTS[split][ix + 1]

    pos_LAX = (
        torch.from_numpy(pos_mmap[r * POS_OFFSET : c * POS_OFFSET])
        .clone()
        .reshape(-1, *META["pos_shape"])
    )

    if unit == "nm":
        pos_LAX /= 10

    pos_LAX = pos_LAX - pos_LAX[:, [1], :].mean(dim=0, keepdim=True)
    # this inserts batch dimension
    pos_BLAX = pos_LAX.unsqueeze(0).repeat(n_repeats, 1, 1, 1)

    if "cuda" in device:
        pos_BLAX = pos_BLAX.pin_memory().to(device, non_blocking=True)
    else:
        pos_BLAX = pos_BLAX.to(device)

    R_BXY = sample_uniform_rotation((n_repeats,), dtype=dtype, device=device)
    pos_BLAX = (
        (R_BXY.unsqueeze(1) @ pos_BLAX.view(n_repeats, -1, 3, 1))
        .squeeze(-1)
        .view(n_repeats, -1, *META["pos_shape"])
    )

    # ca only
    if atoms == "CA":
        pos_BLAX = pos_BLAX[:, :, [1], :].view(n_repeats, -1, 3)
    elif atoms == "all":
        pass
    else:
        raise ValueError(f"Invalid atoms: {atoms}")

    return pos_BLAX


def get_lr_full(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def js_distance(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        return (a * (a.clamp_min(1e-12) / b.clamp_min(1e-12)).log2()).sum()

    return float(
        torch.sqrt(0.5 * (kl(p, m) + kl(q, m)))
    )  # in bits^0.5 (often reported as sqrt(JS))


def codebook_metrics(idx_BL, vocab_size: int):
    counts = torch.bincount(idx_BL.reshape(-1), minlength=vocab_size).float()
    probs = counts / counts.sum()
    entropy = -(probs * probs.clamp_min(1e-12).log2()).sum()
    perplexity = 2**entropy

    # pairwise JS -- do we learn a rotationall invariant encoding?
    # pick a random pair from batch
    B = idx_BL.size(0)
    a, b = int(random.uniform(0, 1) * B), int(random.uniform(0, 1) * B)
    counts_a = torch.bincount(idx_BL[a].reshape(-1), minlength=vocab_size).float()
    counts_b = torch.bincount(idx_BL[b].reshape(-1), minlength=vocab_size).float()
    pa, pb = counts_a / counts_a.sum(), counts_b / counts_b.sum()

    js = js_distance(pa, pb)

    return {
        "entropy": entropy,
        "perplexity": perplexity,
        "js_distance": js,
    }


################################################################################
#                                 training loop                                #
################################################################################


@dataclass
class TrainConfig:
    batch_size = 32
    n_layers_encoder = 12
    n_layers_decoder = 12
    n_heads = 8
    n_channels_decoder = 512
    n_channels_encoder = 512
    n_channels_pair = 64
    levels = (8, 8, 8, 8)
    use_qknorm = False
    use_pair = False
    units = "nm"
    mlp_factor = 4
    learning_rate = 3e-4
    warmup_iters = 1000
    lr_decay_iters = 100000
    min_lr = 1e-4
    dropout = 0.1
    grad_clip = 1.0
    eval_every = 250
    save_every = 1000
    eval_iters = 100
    window_size = 16
    n_micro_steps = 8  # grad accumulation steps
    sigma = 0.0
    use_wandb = True
    descr = "default"
    host = socket.gethostname()
    cb_jitter: float = 0.0
    rot_augment: bool = False
    encoder_type: str = None
    gpt_prior: bool = False
    gpt_weight: float = 0.0
    gpt_prior_start_iter: int = 10_000
    n_neighbors: int = 32


if __name__ == "__main__":
    # global namespace hack is still fun
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.set_default_dtype(torch.float32)
    # I get nan values when using flex attention with bfloat16, so let's not do that.
    # model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    # for debugging -- not sure how to get rotary embeddings to work with bfloat16
    model_dtype = torch.float32

    args = sys.argv[1:]
    cfg = TrainConfig()
    for arg in args:
        if arg.startswith("--"):
            k, v = arg[2:].split("=")
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown argument: {k}")
            print(f"Overriding {k} from {getattr(cfg, k)} to {v}")
            try:
                exec(f"cfg.{k} = {v}")
            except NameError:
                exec(f"cfg.{k} = '{v}'")
    if (not cfg.gpt_prior) and cfg.gpt_weight > 0:
        raise ValueError("gpt_weight > 0 but gpt_prior is False. This is probably a mistake.")
    if cfg.gpt_prior and cfg.gpt_weight == 0:
        raise ValueError("gpt_weight == 0 but gpt_prior is True. This is probably a mistake.")
    #########################################################################3
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    master_process = rank == 0
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    random.seed(1137 + rank)
    torch.manual_seed(1137 + rank)

    if master_process:
        wandb.init(
            project="jovian-dae",
            config=asdict(cfg),
            mode="disabled" if not cfg.use_wandb else "online",
        )
        run_name = wandb.run.name

    run_name_list = [run_name] if master_process else [None]
    if ddp:
        dist.broadcast_object_list(run_name_list, src=0)
    run_name = run_name_list[0]  # now everyone has the same string

    if master_process:
        wandb.run.name = f"{run_name}__{cfg.descr}__{cfg.host}"
        Path(f"checkpoints/{run_name}").mkdir(exist_ok=True, parents=True)
        Path(f"samples/{run_name}").mkdir(exist_ok=True, parents=True)

    model_cfg = DAEConfig(
        n_channels_decoder=cfg.n_channels_decoder,
        n_channels_encoder=cfg.n_channels_encoder,
        n_layers_encoder=cfg.n_layers_encoder,
        n_layers_decoder=cfg.n_layers_decoder,
        n_heads=cfg.n_heads,
        mlp_factor=cfg.mlp_factor,
        sigma=cfg.sigma,
        use_qknorm=cfg.use_qknorm,
        n_channels_pair=cfg.n_channels_pair,
        levels=cfg.levels,
        rot_augment=cfg.rot_augment,
        cb_jitter=cfg.cb_jitter,
        encoder_type = cfg.encoder_type,
        gpt_prior = cfg.gpt_prior,
        window_size = cfg.window_size,
        n_neighbors = cfg.n_neighbors,
    )

    get_batch = partial(
        load_from_mmap,
        atoms="CA",
        n_repeats=cfg.batch_size,
        device=device,
        unit=cfg.units,
    )

    model = DAE(model_cfg).eval().to(device)
    from tqdm import tqdm

    all_idx = []
    with torch.no_grad():
        for _ in tqdm(range(1_000)):
            x_BLD = load_from_mmap("test", "CA", n_repeats=1, device=device)
            *_, idx = model.encode(x_BLD)
            all_idx.append(idx.squeeze())

    seen = set()
    for el in all_idx:
        for i in el:
            seen.add(i.item())
    model = model.train()
    print(len(seen))

    if master_process:
        print(f"Number of parameters: {model.num_params():_}")
    if ddp:
        model = DDP(model, device_ids=[rank], gradient_as_bucket_view=False)

    ############### train loop #################################
    scaler = torch.amp.GradScaler(enabled=(model_dtype == torch.float16))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    get_lr = partial(
        get_lr_full,
        warmup_iters=cfg.warmup_iters,
        learning_rate=cfg.learning_rate,
        lr_decay_iters=cfg.lr_decay_iters,
        min_lr=cfg.min_lr,
    )

    raw_model = model if not ddp else model.module

    # loss estimation function
    @torch.no_grad()
    def estimate_loss():
        losses_return = {}
        seen = set()
        all_idx = []
        hammings = []
        # quickly get utilization over a large number of samples
        for _ in range(5_000):
            x_BLD = load_from_mmap("test", "CA", n_repeats=2, device=device)
            *_, idx = raw_model.encode(x_BLD)
            all_idx.append(idx[0])
            hammings.append((idx[0] == idx[1]).float().mean().item())

        for el in all_idx:
            for i in el:
                seen.add(i.item())

        losses_return["test/cb/utilization"] = len(seen) / prod(cfg.levels)
        losses_return["test/cb/hamming"] = np.mean(hammings)

        for split in ["train", "test"]:
            losses = defaultdict(float)
            for _ in range(cfg.eval_iters):
                batch = get_batch(split)
                idx_BL, loss = model(batch)
                # cb_metrics = codebook_metrics(idx_BL, prod(cfg.levels))

                for k, v in loss.items():
                    losses[k] += v
                # for k, v in cb_metrics.items():
                # losses_return[f'cb/{k}'] += v

            for k in losses:
                losses_return[f"{split}/{k}"] = losses[k] / cfg.eval_iters
        

        return losses_return

    it = 0
    batch = get_batch("train")
    best_val_loss = float("inf")
    model.train()
    while True:
        lr = get_lr(it)

        gpt_weight = cfg.gpt_weight if it >= cfg.gpt_prior_start_iter else 0.0

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % cfg.eval_every == 0 and master_process:
            model.eval()
            with torch.no_grad():
                to_log = estimate_loss()  # we'll use this later
            log_msg = "Evaluation: "
            for k, v in to_log.items():
                if "!" not in k:
                    log_msg += f"[{k}]: {v:.6f} "
            print(log_msg)
            wandb.log(to_log)
            model.train()

        # grad accumulate
        for micro_step in range(cfg.n_micro_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == (cfg.n_micro_steps - 1)
            with torch.autocast(device_type="cuda", dtype=model_dtype):
                _, losses = model(batch)
                # gpt_weight will go to 0
                loss = losses["flow_loss"] + gpt_weight * losses["gpt_prior_loss"]
                loss = loss / cfg.n_micro_steps  # average loss over micro steps


            batch = get_batch("train")
            if loss.isnan().any():
                breakpoint()
            scaler.scale(loss).backward()

        for n, p in model.named_parameters():
            # if (p.grad is not None) and p.grad.norm() < 1.e-3:
            #     print(n)

            if p.grad is None:
                print("None", n)#, p.grad.norm())
                continue


        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # flush gradients quickly

        ############## saving routine ########################
        if (it % cfg.save_every == 0) and master_process and ("dummy" not in run_name):
            # current_val_loss = to_log['test/diff_loss'] +  0.3 * to_log['test/distogram_loss']
            current_val_loss = to_log["test/flow_loss"]
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "it": it,
                "cfg": asdict(cfg),
                "model_cfg": asdict(model_cfg),
                "test_loss": current_val_loss,
            }

            save_best = best_val_loss > current_val_loss
            best_val_loss = min(best_val_loss, current_val_loss)
            if save_best:
                torch.save(ckpt, f"checkpoints/{run_name}/best_model.pt")
            torch.save(ckpt, f"checkpoints/{run_name}/latest_model.pt")

        it += 1
