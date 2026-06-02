import dataclasses
import gc
import json
import logging
import os
import platform
import shutil
import time
from contextlib import nullcontext

import etils.epath as epath
from flax.training import common_utils
import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch._dynamo import OptimizedModule
# FSDP1
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP1,
    ShardingStrategy,
    StateDictType,
)
# FSDP2
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FSDPModule as FSDP2,
    MixedPrecisionPolicy,
)
import tqdm
import wandb
from openpi.models.pi0_config import Pi0Config as Pi0Config_old
import openpi.models_pytorch_new.pi0_config as pi0_config_new
import openpi.models_pytorch_new.pi0 as pi0_new
import openpi.models_pytorch_new.model as pt_model
import openpi.models_pytorch_new.checkpoint_format as ckpt_fmt
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            group="openpi",
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

def setup_distributed(dist_method):
    def get_free_port():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
        return port

    if dist_method == "no_dist":
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        return 1, 0, True, device

    if os.environ.get("MASTER_ADDR", None) is None:
        os.environ["MASTER_ADDR"] = "0.0.0.0"
        os.environ["MASTER_PORT"] = str(get_free_port())
        world_size = int(os.environ.get("WORLD_SIZE", "-1"))
        if world_size == -1:
            os.environ["WORLD_SIZE"] = "1"
            world_size = 1
        rank = int(os.environ.get("RANK", "-1"))
        if rank == -1:
            os.environ["RANK"] = "0"
            rank = 0
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "-1"))
        rank = int(os.environ.get("RANK", "-1"))
        assert world_size != -1 and rank != -1
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
        assert world_size == dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    is_main = (world_size == 1) or (dist.get_rank() == 0)
    return world_size, local_rank, is_main, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def get_base_model(dist_method, model):
    if hasattr(model, '_orig_mod'):
        # torch._dynamo
        model = model._orig_mod
    if dist_method == "no_dist":
        return model
    elif dist_method in ("ddp", "fsdp1"):
        return model.module
    elif dist_method == "fsdp2":
        return model
    else:
        assert False


def apply_ddp(model, world_size, device):
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
        static_graph=world_size >= 8,
    )


def apply_fsdp1(model, device, mixed_precision, use_prefetch):
    auto_wrap_policy = None
    if use_prefetch:
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
        forward_prefetch = True
    else:
        backward_prefetch = None
        forward_prefetch = False

    return FSDP1(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=backward_prefetch,
        forward_prefetch=forward_prefetch,
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
        mixed_precision=mixed_precision,
    )


def apply_fsdp2(model, world_size: int, device, mp_policy, use_noreshard):
    if device.type != "cuda":
        raise RuntimeError("FSDP2 training requires CUDA.")
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()
    mesh = init_device_mesh("cuda", (world_size,))
    reshard_after_forward = not use_noreshard
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)
    return model


@torch.no_grad()
def clip_grad_norm_fsdp2(parameters, max_norm: float, device):
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return torch.tensor(0.0, device=device)

    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    for grad in grads:
        if hasattr(grad, "to_local"):
            local_grad = grad.to_local().detach().float().pow(2).sum().cuda()
            dist.all_reduce(local_grad, op=dist.ReduceOp.SUM)
        else:
            local_grad = grad.detach().float().pow(2).sum().cuda()
        total_sq += local_grad

    total_norm = total_sq.sqrt()
    clip_coef = max_norm / (total_norm.item() + 1e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.mul_(clip_coef)
    return total_norm


@torch.no_grad()
def calc_param_norm(named_parameters, device):
    """Match JAX kernel_params norm: only params with ndim>1, excluding
    embed_tokens, position_embedding, biases, and RMSNorm/LayerNorm scales.

    JAX filter:
      nnx.All(
        nnx.Param,
        nnx.Not(PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
        lambda _, x: x.value.ndim > 1,
      )
    """
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    for name, param in named_parameters.items():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            continue
        if any(x in name for x in ("bias", "scale", "pos_embedding", "embed_tokens", "embedding")):
            continue
        if hasattr(param, "to_local"):
            local_sq = param.to_local().detach().float().pow(2).sum().cuda()
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        else:
            local_sq = param.detach().float().pow(2).sum().cuda()
        total_sq += local_sq
    return total_sq.sqrt()


@torch.no_grad()
def calc_param_std_dict(model, device):
    merge_keys = {
        "PaliGemma.img": "PaliGemma.img",
        "vision_tower": "PaliGemma.img",
        "PaliGemma.llm": "PaliGemma.llm",
        "language_model": "PaliGemma.llm",
        "gemma_expert": "PaliGemma.llm",
        "action_in_proj": "action_in_proj",
        "action_out_proj": "action_out_proj",
        "time_mlp_in": "time_mlp_in",
        "time_mlp_out": "time_mlp_out",
        "llm": "PaliGemma.llm",
        "img": "PaliGemma.img",
    }
    std_dict = {key: 0.0 for key in set(merge_keys.values())}

    for k, param in model.named_parameters():
        if hasattr(param, "to_local"):
            std = param.to_local().detach().float().pow(2).sum().cuda()
            dist.all_reduce(std, op=dist.ReduceOp.SUM)
        else:
            std = param.detach().float().pow(2).sum()

        for merge_k, merge_v in merge_keys.items():
            if merge_k in k:
                std_dict[merge_v] += std.item()
                break
        else:
            std_dict[k] = std.item()

    return std_dict

def init_model(
    config: _config.TrainConfig,
    dist_method,
    device,
    world_size,
    is_main,
    compile_mode,
    use_autocast,
):
    model_load_dtype = config.pytorch_training_precision
    mp_policy = None
    if model_load_dtype == "mp_bfloat16":
        model_load_dtype = "float32"
        if dist_method in ["no_dist", "ddp"]:
            assert use_autocast == True
        elif dist_method == "fsdp1":
            if not use_autocast:
                from torch.distributed.fsdp import MixedPrecision
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                )
        elif dist_method == "fsdp2":
            if not use_autocast:
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                    cast_forward_inputs=True,
                )
        else:
            assert False
    else:
        assert use_autocast == False

    assert isinstance(config.model, Pi0Config_old)
    model_cfg = pi0_config_new.Pi0Config(**config.model.__dict__)

    model = pi0_new.Pi0(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if is_main:
            logging.info("Enabled gradient checkpointing")

    # Load weights
    if config.pytorch_weight_path is not None:
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No weights found at {model_path}")
        state_dict = safetensors.torch.load_file(model_path)
        if config.pytorch_dist_args.get("load_old_ckpt_fmt", False):
            state_dict = ckpt_fmt.old_to_new_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        if is_main:
            logging.info(f"Loaded weights from {model_path}")

    if model_load_dtype == "float32":
        model.to(dtype=torch.float32)
        if is_main:
            logging.info("Converted all params to float32")
    elif model_load_dtype == "bfloat16":
        model.to(dtype=torch.bfloat16)
        if is_main:
            logging.info("Converted all params to bfloat16")
    else:
        assert False

    compile_position = None
    if compile_mode != "null":
        if dist_method not in ["fsdp1", "fsdp2"]:
            compile_position = "before"
        else:
            compile_position = "after"
        assert compile_position in ["before", "after"]

    if compile_position == "before":
        model = torch.compile(model, mode=compile_mode, dynamic=False)
        if is_main:
            logging.info(f"Enable torch.compile(mode={compile_mode}) BEFORE")

    if dist_method == "ddp":
        model = apply_ddp(model, world_size, device)
    elif dist_method == "fsdp1":
        use_prefetch = bool(int(os.environ.get("USE_PREFETCH", "0")))
        model = apply_fsdp1(model, device, mixed_precision=mp_policy, use_prefetch=use_prefetch)
        if is_main:
            logging.info(f"Enabled FSDP1 with FULL_SHARD and auto-wrap policy, use_prefetch={use_prefetch}")
    elif dist_method == "fsdp2":
        use_noreshard = bool(int(os.environ.get("USE_NORESHARD", "0")))
        model = apply_fsdp2(model, world_size, device, mp_policy=mp_policy, use_noreshard=use_noreshard)
        if is_main:
            logging.info(f"Enabled FSDP2 with fully_shard, use_noreshard={use_noreshard}")

    if compile_position == "after":
        model = torch.compile(model, mode=compile_mode, dynamic=False)
        if is_main:
            logging.info(f"Enable torch.compile(mode={compile_mode}) After")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr
    optim_kwargs = dict(
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
        fused=True,
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        **optim_kwargs,
    )

    if is_main:
        optim_dtypes = set()
        for param_group in optim.param_groups:
            for p in param_group["params"]:
                optim_dtypes.add(p.dtype)
        logging.info(f"Training precision: {model_cfg.dtype}")
        logging.info(f"optim_dtypes: {optim_dtypes}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
    return model, optim


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig, use_consistent):
    data_loader = _data_loader.create_behavior_data_loader_torch(
        config,
        skip_norm_stats=False,
        shuffle=not use_consistent,
    )
    return data_loader


def _send_batches_to_rank(batches, dst_rank: int):
    """Send K batches to dst_rank via send_object_list."""
    dist.send_object_list([batches], dst=dst_rank)


def _recv_batches_from_rank(src_rank: int) -> list:
    """Receive K batches from src_rank via recv_object_list."""
    obj_list = [None]
    dist.recv_object_list(obj_list, src=src_rank)
    return obj_list[0]


def _make_rngs_per_sample(global_step: int, config: _config.TrainConfig) -> list[torch.Generator]:
    """Build per-sample CPU RNGs for one step."""
    K = config.gradient_accumulate
    W = dist.get_world_size() if dist.is_initialized() else 1
    rank_id = dist.get_rank() if dist.is_initialized() else 0
    lb = config.batch_size // W
    rngs = []
    for _s in range(K * lb):
        _mi = _s // lb
        _p = _s % lb
        _logical_idx = (global_step * K * lb + _mi * lb + _p) * W + rank_id
        _seed = (config.seed * 1000003 + int(_logical_idx)) & ((1 << 63) - 1)
        rngs.append(torch.Generator(device="cpu").manual_seed(_seed))
    return rngs


def _make_noise_time(rngs: list[torch.Generator], micro_idx: int, config: _config.TrainConfig, dtype: torch.dtype, device: torch.device):
    """Build per-sample noise and flow-matching time for one micro-batch."""
    K = config.gradient_accumulate
    W = dist.get_world_size() if dist.is_initialized() else 1
    lb = config.batch_size // W
    AH = config.model.action_horizon
    AD = config.model.action_dim
    B_local = config.batch_size // max(W, 1)
    beta_conc = torch.tensor([1.5, 1.0])
    noise_list, time_list = [], []
    for _p in range(B_local):
        g = rngs[micro_idx * lb + _p]
        noise_list.append(torch.randn((AH, AD), generator=g))
        t_dir = torch._sample_dirichlet(beta_conc, generator=g)
        time_list.append(t_dir[0])
    noise = torch.stack(noise_list, dim=0).to(dtype=dtype, device=device)
    time = torch.stack(time_list, dim=0).to(dtype=dtype, device=device)
    time = time * 0.999 + 0.001
    return noise, time


def train_step(
    config: _config.TrainConfig,
    dist_method,
    model,
    optim,
    device,
    global_step,
    batches,
    use_autocast,
    use_consistent,
):
    def lr_schedule(step: int):
        warmup_steps = config.lr_schedule.warmup_steps
        peak_lr = config.lr_schedule.peak_lr
        decay_steps = config.lr_schedule.decay_steps
        end_lr = config.lr_schedule.decay_lr

        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # Update LR
    for pg in optim.param_groups:
        pg["lr"] = lr_schedule(global_step)

    if use_consistent:
        rngs_per_sample = _make_rngs_per_sample(global_step, config)

    loss_acc = []
    for micro_idx, batch in enumerate(batches):
        observation, actions = batch
        observation = _move_to_device(observation, device)
        actions = actions.to(device=device)

        noise, time = None, None
        if use_consistent:
            noise, time = _make_noise_time(rngs_per_sample, micro_idx, config, actions.dtype, device)

        # Forward pass
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
        with amp_ctx:
            losses = model(observation, actions, train=True, noise=noise, time=time)
        if not isinstance(losses, torch.Tensor):
            losses = torch.tensor(losses, device=device, dtype=torch.float32)

        loss = losses.mean() / len(batches)
        loss_acc.append(loss.detach())
        loss.backward()

    loss_acc = torch.stack(loss_acc).sum().cuda()
    dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)

    # Gradient clipping
    if dist_method in ["no_dist", "ddp"]:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
    elif dist_method == "fsdp1":
        grad_norm = model.clip_grad_norm_(max_norm=config.optimizer.clip_gradient_norm)
    elif dist_method == "fsdp2":
        grad_norm = clip_grad_norm_fsdp2(model.parameters(), config.optimizer.clip_gradient_norm, device)
    else:
        assert False

    # Optimizer step
    optim.step()
    optim.zero_grad(set_to_none=True)

    trainable_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    param_norm = calc_param_norm(trainable_params, device)
    info = {
        "loss": loss_acc.item(),
        "learning_rate": optim.param_groups[0]["lr"],
        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "param_norm": float(param_norm),
    }
    return info


def save_checkpoint(dist_method, model, optimizer, global_step, config: _config.TrainConfig, is_main, data_config):
    should_save = (global_step % config.save_interval == 0 and global_step > 0) or (
        global_step == config.num_train_steps - 1
    )
    if not should_save:
        return

    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

    if is_main:
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # model.state_dict() returns master params (if wrapper) or normal state dict
    if dist_method in ["no_dist", "ddp"]:
        if is_main:
            base = get_base_model(dist_method, model)
            state_dict = base.state_dict()
            # Clone tied weights (embed_tokens/lm_head share storage) for safetensors compat
            for k in list(state_dict.keys()):
                for k2 in state_dict:
                    if k2 != k and state_dict[k].untyped_storage().data_ptr() == state_dict[k2].untyped_storage().data_ptr():
                        state_dict[k] = state_dict[k].clone()
            safetensors.torch.save_file(state_dict, tmp_ckpt_dir / "model.safetensors")
            torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")
    elif dist_method == "fsdp1":
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP1.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config, optim_config):
            model_state = model.state_dict()
            optimizer_state = FSDP1.optim_state_dict(model, optimizer)
        if is_main:
            torch.save(model_state, tmp_ckpt_dir / "model.pt")
            torch.save(optimizer_state, tmp_ckpt_dir / "optimizer.pt")
    elif dist_method == "fsdp2":
        state_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state, optimizer_state = get_state_dict(model, optimizer, options=state_options)
        if is_main:
            torch.save(model_state, tmp_ckpt_dir / "model.pt")
            torch.save(optimizer_state, tmp_ckpt_dir / "optimizer.pt")
    else:
        assert False

    if is_main:
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)

    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(dist_method, model, optimizer, checkpoint_dir, device, is_main):
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    if is_main:
        logging.info("Loading model state...")
    safetensors_path = ckpt_dir / "model.safetensors"
    torch_model_path = ckpt_dir / "model.pt"

    loaded = False
    if dist_method == "no_dist":
        if safetensors_path.exists():
            safetensors.torch.load_model(get_base_model(dist_method, model), safetensors_path, device=str(device))
            model_state_dict = None
            loaded = True
    elif dist_method == "fsdp1":
        if safetensors_path.exists():
            with FSDP.summon_full_params(model, writeback=True, rank0_only=False):
                safetensors.torch.load_model(get_base_model(dist_method, model), safetensors_path, device=str(device))
            loaded = True
    elif dist_method == "fsdp2":
        if torch_model_path.exists():
            model_state_dict = torch.load(torch_model_path, map_location="cpu", weights_only=False)
            loaded = True
    else:
        assert False
    if not loaded:
        raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

    torch.cuda.empty_cache()
    gc.collect()
    if is_main:
        log_memory_usage(device, latest_step, "after_loading_model")

    logging.info("Loading optimizer state...")
    optimizer_path = ckpt_dir / "optimizer.pt"
    if optimizer_path.exists():
        optimizer_state_dict = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

    if dist_method in ["no_dist", "ddp"]:
        optimizer.load_state_dict(optimizer_state_dict)
    elif dist_method == "fsdp1":
        optimizer_state_dict = FSDP1.optim_state_dict_to_load(model, optimizer, optimizer_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    elif dist_method == "fsdp2":
        state_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optimizer_state_dict,
            options=state_options,
        )
    else:
        assert False

    del model_state_dict
    del optimizer_state_dict
    torch.cuda.empty_cache()
    gc.collect()
    if is_main:
        log_memory_usage(device, latest_step, "after_loading_optimizer")

    # Load metadata
    logging.info("Loading metadata...")
    metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
    global_step = metadata.get("global_step", latest_step)
    del metadata
    torch.cuda.empty_cache()
    gc.collect()
    if is_main:
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
    return global_step


def get_latest_checkpoint_step(checkpoint_dir):
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    dist_info = ""
    if dist.is_initialized():
        dist_info = f" | distributed: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{dist_info}"
    )


def _move_to_device(obj, device):
    """Recursively move nested dict/list/tuple/tensor to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    if hasattr(obj, 'images') and hasattr(obj, 'state'):
        return pt_model.Observation(
            images=_move_to_device(obj.images, device),
            image_masks=_move_to_device(obj.image_masks, device),
            state=obj.state.to(device),
            tokenized_prompt=obj.tokenized_prompt.to(device) if obj.tokenized_prompt is not None else None,
            tokenized_prompt_mask=obj.tokenized_prompt_mask.to(device) if obj.tokenized_prompt_mask is not None else None,
            token_ar_mask=obj.token_ar_mask.to(device) if obj.token_ar_mask is not None else None,
            token_loss_mask=obj.token_loss_mask.to(device) if obj.token_loss_mask is not None else None,
            pcd_xyz=obj.pcd_xyz.to(device) if obj.pcd_xyz is not None else None,
        )
    return obj



def main(config: _config.TrainConfig):
    dist_method = os.environ.get("DIST_METHOD", "")
    if dist_method == "":
        dist_method = config.pytorch_dist_method
    assert dist_method in ["no_dist", "ddp", "fsdp1", "fsdp2"]
    if dist_method == "no_dist":
        assert os.environ.get("MASTER_ADDR", None) is None

    compile_mode = os.environ.get("TORCH_COMPILE_MODE", "")
    if compile_mode == "":
        compile_mode = config.pytorch_dist_args.get(
            "torch_compile_mode", "default"
        )

    use_autocast = os.environ.get("USE_AUTOCAST", "")
    if use_autocast != "":
        use_autocast = bool(int(use_autocast))
    else:
        use_autocast = config.pytorch_dist_args.get(
            "use_autocast", False
        )

    use_consistent = os.environ.get("USE_CONSISTENT", "")
    if use_consistent != "":
        use_consistent = bool(int(use_consistent))
    else:
        use_consistent = config.pytorch_dist_args.get(
            "use_consistent", False
        )

    if config.batch_size % torch.cuda.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {torch.cuda.device_count()}."
        )

    world_size, local_rank, is_main, device = setup_distributed(dist_method)
    set_seed(config.seed, local_rank)

    if is_main:
        init_logging()
        logging.info(f"Running on: {platform.node()}")
        logging.info(f"PT global device count: {torch.cuda.device_count()}")
        logging.info(f"Dist method: {dist_method}")

    if True:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if is_main:
            logging.info("Enabled cudnn benchmark and TF32 optimizations")

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                if is_main:
                    logging.info(
                        f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                    )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
            logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        if is_main:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created checkpoint directory: {config.checkpoint_dir}")
    else:
        if is_main:
            logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    effective_batch_size = config.batch_size // world_size
    if is_main:
        logging.info(
            f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
        )

    # Only rank 0 reads from dataloader; other ranks receive batches via
    # broadcast_object_list at each step. data_config is sent at save time.
    if is_main:
        data_loader = build_datasets(config, use_consistent)
        data_config = data_loader.data_config()
        data_iter = iter(data_loader)
    else:
        data_config = None

    model, optim = init_model(
        config,
        dist_method,
        device,
        world_size,
        is_main,
        compile_mode,
        use_autocast,
    )
    if dist_method == "fsdp1":
        _m = model._orig_mod if isinstance(model, OptimizedModule) else model
        assert isinstance(_m, FSDP1)
    elif dist_method == "fsdp2":
        assert isinstance(_m, FSDP2)

    # Load checkpoint if resuming
    start_step = 0
    if resuming:
        start_step = load_checkpoint(dist_method, model, optim, config.checkpoint_dir, device, is_main)
        if is_main:
            logging.info(f"Resumed training from step {start_step}")

    model.train()

    # CUDA graph warmup:
    # if compile_mode is not None:
    #     WARMUP_STEPS = 3
    #     if is_main:
    #         logging.info(f"Running warmup ({WARMUP_STEPS} steps)...")
    #     warmup_loader = debug_data_loader(effective_batch_size)
    #     warmup_iter = itertools.cycle(warmup_loader)
    #     warmup_amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
    #     for _ in range(WARMUP_STEPS):
    #         observation, actions = next(warmup_iter)
    #         observation = _move_to_device(observation, device)
    #         actions = actions.to(device=device)
    #         with warmup_amp_ctx:
    #             losses = model(observation, actions, train=True)
    #         if not isinstance(losses, torch.Tensor):
    #             losses = torch.tensor(losses, device=device, dtype=torch.float32)
    #         loss = losses.mean()
    #         loss.backward()
    #         optim.zero_grad(set_to_none=True)
    #     del observation, actions, losses, loss
    #     if is_main:
    #         logging.info("CUDA graph warmup complete.")

    if is_main:
        log_memory_usage(device, 0, "after_model_init")
        logging.info(
            f"Running on: {platform.node()} | world_size={world_size}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")

    param_std_dict = calc_param_std_dict(model, device)
    if is_main:
        logging.info(f"Initialized param_std_dict: {json.dumps(param_std_dict, indent=2)}")
        logging.info(f"Initialized param_norm: {sum(param_std_dict.values())**0.5}")

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    t_start = time.perf_counter()

    for global_step in pbar:
        if is_main and hasattr(data_loader, "set_epoch"):
            data_loader.set_epoch(global_step // len(data_loader))

        # Rank 0 reads K micro-batches for each rank and sends them.
        if is_main:
            for r in range(1, world_size):
                _send_batches_to_rank(
                    [next(data_iter) for _ in range(config.gradient_accumulate)],
                    dst_rank=r,
                )
            batches = [next(data_iter) for _ in range(config.gradient_accumulate)]
        else:
            batches = _recv_batches_from_rank(src_rank=0)

        info = train_step(
            config,
            dist_method,
            model,
            optim,
            device,
            global_step,
            batches,
            use_autocast=use_autocast,
            use_consistent=use_consistent,
        )
        infos.append(info)

        step_time = time.perf_counter() - t_start
        t_start = time.perf_counter()
        if global_step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.tree.map(np.mean, stacked_infos)
            reduced_info = {k: reduced_info[k] for k in sorted(reduced_info.keys())}
            info_str = ", ".join(f"{k}={v}" for k, v in reduced_info.items()) + f", step_time={step_time:.2f}s"
            if is_main:
                pbar.write(f"Step {global_step}: {info_str}")
                pbar.set_postfix(reduced_info)

            if is_main and config.wandb_enabled:
                wandb.log(reduced_info, step=global_step)
            infos = []

        should_save = (
            (global_step % config.save_interval == 0 and global_step > start_step)
            or global_step == config.num_train_steps - 1
        )
        if should_save:
            save_checkpoint(dist_method, model, optim, global_step, config, is_main, data_config)

    time_now = time.time()
    time_all = time_now - pbar.start_t
    time_step = time_all / (config.num_train_steps - start_step)
    if is_main:
        pbar.write(f"All time: {time_all}, Step time: {time_step}")
    pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_distributed()

def debug_data_loader(batch_size):
    from openpi.models.model import Observation
    import numpy as np
    fake_rng = np.random.default_rng(42)
    fake_batch_dict = {
        'image': {},
        'image_mask': {},
        'state': fake_rng.normal(0, 1, (batch_size, 32)),
        'tokenized_prompt': fake_rng.integers(0, 100, (batch_size, 200), dtype=np.int64),
        'tokenized_prompt_mask': np.ones((batch_size, 200), dtype=np.bool_),
        'actions': fake_rng.normal(0, 1, (batch_size, 32, 32)),
    }
    for k in ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
        fake_batch_dict['image'][k] = fake_rng.integers(0, 256, (batch_size, 224, 224, 3), dtype=np.uint8)
        fake_batch_dict['image_mask'][k] = np.ones((batch_size,), dtype=np.bool_)

    fake_batch_dict = jax.tree.map(torch.as_tensor, fake_batch_dict)
    fake_obs = Observation.from_dict(fake_batch_dict)
    fake_action = fake_batch_dict['actions']
    return [(fake_obs, fake_action)]


if __name__ == "__main__":
    main(_config.cli())
