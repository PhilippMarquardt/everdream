from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.distributed as dist

from everdream.common import COMPUTE_DTYPE, get_dist_info, get_peak_flops, print0
from everdream.config.schema import EverdreamConfig
from everdream.data.dataloader import tokenizing_weighted_data_loader_bos_bestfit
from everdream.data.sources import ensure_dataset_ready
from everdream.eval import run_eval
from everdream.logging.wandb_logger import init_wandb
from everdream.models.registry import build_model
from everdream.tokenizer import get_tokenizer


def _safe_runtime_config(runtime_cfg):
    runtime_dict = runtime_cfg.__dict__.copy()
    if "hf_token" in runtime_dict and runtime_dict["hf_token"]:
        runtime_dict["hf_token"] = "***"
    if "wandb_api_key" in runtime_dict and runtime_dict["wandb_api_key"]:
        runtime_dict["wandb_api_key"] = "***"
    return runtime_dict


def _normalize_output(out):
    if isinstance(out, torch.Tensor):
        return {'logits': None, 'total': out, 'ce': out, 'lb': out.new_zeros(()), 'rz': out.new_zeros(())}
    if isinstance(out, tuple):
        if len(out) == 2:
            logits, loss = out
            if loss is None:
                return {'logits': logits, 'total': None, 'ce': None, 'lb': None, 'rz': None}
            return {'logits': logits, 'total': loss, 'ce': loss, 'lb': loss.new_zeros(()), 'rz': loss.new_zeros(())}
        if len(out) == 5:
            logits, total, ce, lb, rz = out
            return {'logits': logits, 'total': total, 'ce': ce, 'lb': lb, 'rz': rz}
    raise TypeError(f'Unsupported model output type: {type(out)}')


def _unwrap_state_dict(state_dict):
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def _checkpoint_paths(base_dir: str, step: int, rank: int):
    return {
        "model": os.path.join(base_dir, f"model_{step:06d}.pt"),
        "optim": os.path.join(base_dir, f"optim_{step:06d}_rank{rank:d}.pt"),
        "meta": os.path.join(base_dir, f"meta_{step:06d}.json"),
    }


def _save_checkpoint(base_dir: str, step: int, model, optimizer, cfg: EverdreamConfig, rank: int, master_process: bool):
    os.makedirs(base_dir, exist_ok=True)
    paths = _checkpoint_paths(base_dir, step, rank)
    target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if master_process:
        torch.save(_unwrap_state_dict(target_model.state_dict()), paths["model"])
        meta = {
            "step": step,
            "runtime": _safe_runtime_config(cfg.runtime),
            "tokenizer": cfg.tokenizer.__dict__,
            "training": cfg.training.__dict__,
            "model": cfg.model,
            "datasets": [d.__dict__ for d in cfg.datasets],
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    torch.save(optimizer.state_dict(), paths["optim"])
    if dist.is_initialized():
        dist.barrier()


def _load_checkpoint(base_dir: str, step: int, model, optimizer, device, rank: int):
    paths = _checkpoint_paths(base_dir, step, rank)
    model_state = torch.load(paths["model"], map_location=device)
    target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    target_model.load_state_dict(model_state, strict=True)
    optim_state = torch.load(paths["optim"], map_location=device)
    optimizer.load_state_dict(optim_state)
    meta = None
    if os.path.exists(paths["meta"]):
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    if dist.is_initialized():
        dist.barrier()
    return meta


def _maybe_enable_fp8(model, runtime_cfg, device_type):
    if not runtime_cfg.fp8:
        return model
    if device_type != 'cuda':
        print0('Warning: FP8 requested without CUDA, ignoring')
        return model
    from everdream.fp8 import Float8LinearConfig, convert_to_float8_training
    import torch.nn as nn

    def fp8_module_filter(mod: nn.Module, _fqn: str) -> bool:
        return isinstance(mod, nn.Linear) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0 and min(mod.in_features, mod.out_features) >= 128

    fp8_config = Float8LinearConfig.from_recipe_name(runtime_cfg.fp8_recipe)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
    num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
    print0(f'FP8 training enabled ({runtime_cfg.fp8_recipe}) - converted {num_fp8}/{num_linear} linear layers')
    return model


def train(cfg: EverdreamConfig, device, master_process: bool = True):
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    if master_process:
        for spec in cfg.datasets:
            ensure_dataset_ready(spec)
    if dist.is_initialized():
        dist.barrier()
    tokenizer = get_tokenizer(
        tokenizer_dir=cfg.tokenizer.path,
        source=cfg.tokenizer.source,
        ensure_chat_special_tokens=cfg.tokenizer.ensure_chat_special_tokens,
    )
    vocab_size = tokenizer.get_vocab_size()
    model = build_model(cfg.model, vocab_size=vocab_size, sequence_len=cfg.training.max_seq_len, runtime_cfg=cfg.runtime)
    model = model.to(device)
    if hasattr(model, 'init_weights'):
        model.init_weights()
    model = _maybe_enable_fp8(model, cfg.runtime, device.type)
    if cfg.runtime.compile and hasattr(torch, 'compile'):
        model = torch.compile(model, dynamic=False)
    optimizer = model.setup_optimizer(
        unembedding_lr=cfg.training.unembedding_lr,
        embedding_lr=cfg.training.embedding_lr,
        matrix_lr=cfg.training.matrix_lr,
        weight_decay=cfg.training.weight_decay,
        adam_betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        scalar_lr=cfg.training.scalar_lr,
    )

    if cfg.runtime.checkpoint_path and cfg.training.resume_from_step > 0:
        meta = _load_checkpoint(cfg.runtime.checkpoint_path, cfg.training.resume_from_step, model, optimizer, device, ddp_rank)
        if master_process:
            print0(f"Resumed from step {cfg.training.resume_from_step}")
            if meta is not None:
                print0(f"Checkpoint path: {cfg.runtime.checkpoint_path}")

    run = init_wandb(
        enabled=cfg.runtime.wandb,
        project=cfg.runtime.wandb_project,
        run_name=cfg.runtime.run_name,
        config={'runtime': _safe_runtime_config(cfg.runtime), 'tokenizer': cfg.tokenizer.__dict__, 'training': cfg.training.__dict__, 'model': cfg.model, 'datasets': [d.__dict__ for d in cfg.datasets]},
        master_process=master_process,
        mode=cfg.runtime.wandb_mode,
    )

    train_loader = tokenizing_weighted_data_loader_bos_bestfit(tokenizer, cfg.datasets, B=cfg.training.device_batch_size, T=cfg.training.max_seq_len, split='train', seed=cfg.runtime.seed, device=device.type)
    val_loader = tokenizing_weighted_data_loader_bos_bestfit(tokenizer, cfg.datasets, B=cfg.training.device_batch_size, T=cfg.training.max_seq_len, split='val', seed=cfg.runtime.seed, device=device.type)

    num_flops_per_token = model.estimate_flops()
    local_microbatch_tokens = cfg.training.device_batch_size * cfg.training.max_seq_len
    global_microbatch_tokens = local_microbatch_tokens * ddp_world_size
    if cfg.training.total_batch_size % global_microbatch_tokens != 0:
        raise ValueError(
            f"total_batch_size={cfg.training.total_batch_size} must be divisible by "
            f"device_batch_size*max_seq_len*world_size={global_microbatch_tokens}"
        )
    grad_accum = max(1, cfg.training.total_batch_size // global_microbatch_tokens)
    tokens_per_step = global_microbatch_tokens * grad_accum
    flops_per_step = num_flops_per_token * tokens_per_step
    active_params = model.num_scaling_params()['active']
    if cfg.training.num_iterations > 0:
        num_iterations = cfg.training.num_iterations
    elif cfg.training.target_tokens > 0:
        num_iterations = max(1, round(cfg.training.target_tokens / tokens_per_step))
    elif cfg.training.target_flops > 0:
        num_iterations = round(cfg.training.target_flops / flops_per_step)
    else:
        target_tokens = active_params * cfg.training.target_param_data_ratio
        num_iterations = max(1, round(target_tokens / tokens_per_step))

    # Weight decay scaling (ref: Nanochat T_epoch framework, arxiv 2405.13698)
    B_REF = 524288
    D_REF = 10.5 * (12 * (4 * 768**2 + 2 * 768 * 4 * 768) + 768 * vocab_size)
    total_tokens = num_iterations * tokens_per_step
    scaled_wd = cfg.training.weight_decay * math.sqrt(tokens_per_step / B_REF) * (D_REF / total_tokens)

    print0(f'Train iterations: {num_iterations} | Active params: {active_params:,}')
    print0(f'Tokens/step: {tokens_per_step:,} | Target tokens: {num_iterations * tokens_per_step:,}')
    print0(f'FLOPs/token: {num_flops_per_token:.2e} | FLOPs/step: {flops_per_step:.2e}')
    print0(f'Weight decay scaled: {cfg.training.weight_decay} -> {scaled_wd:.6f}')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print0(f'GPU: {gpu_name} | Peak FLOPS (BF16): {get_peak_flops(gpu_name):.2e}')

    def get_lr_multiplier(step):
        warmup_iters = round(cfg.training.warmup_ratio * num_iterations)
        warmdown_iters = round(cfg.training.warmdown_ratio * num_iterations)
        if warmup_iters > 0 and step < warmup_iters:
            return step / warmup_iters
        if step <= num_iterations - warmdown_iters:
            return 1.0
        frac = (num_iterations - step) / max(1, warmdown_iters)
        return cfg.training.final_lr_frac + (1 - cfg.training.final_lr_frac) * frac

    @torch.no_grad()
    def eval_loss():
        model.eval()
        totals = {'total': 0.0, 'ce': 0.0, 'lb': 0.0, 'rz': 0.0}
        count = max(1, cfg.training.eval_tokens // (cfg.training.device_batch_size * cfg.training.max_seq_len))
        for _ in range(count):
            x, y, _ = next(val_loader)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE):
                out = _normalize_output(model(x, y))
            totals['total'] += out['total'].item()
            totals['ce'] += out['ce'].item()
            totals['lb'] += out['lb'].item()
            totals['rz'] += out['rz'].item()
        if dist.is_initialized():
            t = torch.tensor([totals['total'], totals['ce'], totals['lb'], totals['rz']], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            totals['total'], totals['ce'], totals['lb'], totals['rz'] = t.tolist()
        model.train()
        for key in totals:
            totals[key] /= count
        totals['aux'] = totals['lb'] + totals['rz']
        return totals

    def maybe_run_extended_eval(step: int):
        if not master_process:
            return
        modes = cfg.training.eval_modes
        if not modes:
            return
        if cfg.training.eval_every <= 0 or step % cfg.training.eval_every != 0:
            return
        eval_results = run_eval(
            model=model,
            tokenizer=tokenizer,
            device=device,
            val_loader=val_loader,
            eval_tokens=cfg.training.eval_tokens,
            eval_batch_tokens=cfg.training.device_batch_size * cfg.training.max_seq_len,
            eval_modes=[m for m in modes if m != "core"],
            core_metric_max_per_task=cfg.training.core_metric_max_per_task,
        )
        if "bpb" in eval_results:
            print0(f"  -> val_bpb {eval_results['bpb']:.4f}")
            run.log({'step': step, 'val/bpb': eval_results['bpb']})
        if "samples" in eval_results:
            print0("  -> samples")
            for sample in eval_results["samples"]:
                print0(f"     {sample}")
            sample_payload = {f'sample/{i}': s for i, s in enumerate(eval_results["samples"])}
            run.log({'step': step, **sample_payload})
        if "core" in modes and cfg.training.core_metric_every > 0 and step % cfg.training.core_metric_every == 0:
            core_results = run_eval(
                model=model,
                tokenizer=tokenizer,
                device=device,
                val_loader=val_loader,
                eval_tokens=cfg.training.eval_tokens,
                eval_batch_tokens=cfg.training.device_batch_size * cfg.training.max_seq_len,
                eval_modes=["core"],
                core_metric_max_per_task=cfg.training.core_metric_max_per_task,
            )
            core = core_results.get("core")
            if core is not None:
                print0(f"  -> core_metric {core['core_metric']:.4f}")
                run.log({'step': step, 'val/core_metric': core['core_metric']})

    model.train()
    log_t0 = time.time()
    log_every = cfg.training.log_every
    log_losses = []
    start_step = max(0, cfg.training.resume_from_step)
    for step in range(start_step + 1, num_iterations + 1):
        lrm = get_lr_multiplier(step)
        muon_frac = min(step / 300, 1)
        muon_momentum = (1 - muon_frac) * 0.85 + muon_frac * 0.95
        muon_wd = scaled_wd * (1 - step / num_iterations)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * lrm
            if group.get('kind') == 'muon':
                group['momentum'] = muon_momentum
                group['weight_decay'] = muon_wd
        optimizer.zero_grad(set_to_none=True)
        total_acc = ce_acc = lb_acc = rz_acc = 0.0
        for _ in range(grad_accum):
            x, y, _ = next(train_loader)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE):
                out = _normalize_output(model(x, y))
                (out['total'] / grad_accum).backward()
            total_acc += out['total'].item() / grad_accum
            ce_acc += out['ce'].item() / grad_accum
            lb_acc += out['lb'].item() / grad_accum
            rz_acc += out['rz'].item() / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        log_losses.append({'total': total_acc, 'ce': ce_acc, 'lb': lb_acc, 'rz': rz_acc})
        if step % log_every == 0:
            window = log_losses[-log_every:]
            avg_total = sum(d['total'] for d in window) / len(window)
            avg_ce = sum(d['ce'] for d in window) / len(window)
            avg_lb = sum(d['lb'] for d in window) / len(window)
            avg_rz = sum(d['rz'] for d in window) / len(window)
            avg_aux = avg_lb + avg_rz
            dt = time.time() - log_t0
            local_tps = cfg.training.device_batch_size * grad_accum * (cfg.training.max_seq_len - 1) * log_every / dt
            global_tps = local_tps * ddp_world_size
            print0(f"step {step:5d}/{num_iterations} | total {avg_total:.4f} | ce {avg_ce:.4f} | aux {avg_aux:.4f} | lb {avg_lb:.4f} | rz {avg_rz:.4f} | lrm {lrm:.2f} | {global_tps/1e3:.0f}k tok/s")
            run.log({'step': step, 'train/total': avg_total, 'train/ce': avg_ce, 'train/lb': avg_lb, 'train/rz': avg_rz, 'train/aux': avg_aux, 'train/tok_s_global': global_tps, 'train/tok_s_local': local_tps, 'train/lrm': lrm})
            log_t0 = time.time()
        if cfg.training.eval_every > 0 and step % cfg.training.eval_every == 0:
            metrics = eval_loss()
            ppl = math.exp(metrics['ce'])
            print0(f"  -> val_total {metrics['total']:.4f} | val_ce {metrics['ce']:.4f} | val_aux {metrics['aux']:.4f} | ppl {ppl:.1f}")
            run.log({'step': step, 'val/total': metrics['total'], 'val/ce': metrics['ce'], 'val/lb': metrics['lb'], 'val/rz': metrics['rz'], 'val/aux': metrics['aux'], 'val/ppl': ppl})
            maybe_run_extended_eval(step)
        if cfg.runtime.checkpoint_path and cfg.training.save_every > 0 and step % cfg.training.save_every == 0:
            _save_checkpoint(cfg.runtime.checkpoint_path, step, model, optimizer, cfg, ddp_rank, master_process)
    run.finish()
