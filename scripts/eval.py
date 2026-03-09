
from __future__ import annotations

import argparse
import os

from everdream.common import print_banner
from everdream.config.load import load_config
from everdream.data.sources import ensure_dataset_ready
from everdream.eval import run_eval
from everdream.models.registry import build_model
from everdream.runtime.distributed import compute_init, compute_cleanup, print0
from everdream.runtime.notebook import init_notebook
from everdream.tokenizer import get_tokenizer
import torch


def main():
    parser = argparse.ArgumentParser(description="Standalone evaluation for everdream models")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-tokens", type=int, default=40 * 524288)
    parser.add_argument("--eval", type=str, default="core,bpb,sample")
    parser.add_argument("--checkpoint-step", type=int, default=-1)
    args = parser.parse_args()

    print_banner()
    cfg = load_config(args.config)
    if cfg.runtime.hf_token:
        os.environ["HF_TOKEN"] = cfg.runtime.hf_token
    if cfg.runtime.wandb_api_key:
        os.environ["WANDB_API_KEY"] = cfg.runtime.wandb_api_key
    if cfg.runtime.notebook:
        init_notebook(
            mount_drive=cfg.runtime.mount_drive,
            drive_path=cfg.runtime.drive_path,
            install_gpu_extras=cfg.runtime.install_gpu_extras,
            install_moe=cfg.runtime.install_moe,
            install_hybrid=cfg.runtime.install_hybrid,
        )

    ddp, ddp_rank, _, _, device = compute_init(cfg.runtime.device_type or "")
    try:
        if ddp_rank == 0:
            for spec in cfg.datasets:
                ensure_dataset_ready(spec)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        tokenizer = get_tokenizer(
            tokenizer_dir=cfg.tokenizer.path,
            source=cfg.tokenizer.source,
            ensure_chat_special_tokens=cfg.tokenizer.ensure_chat_special_tokens,
        )
        vocab_size = tokenizer.get_vocab_size()
        model = build_model(cfg.model, vocab_size=vocab_size, sequence_len=cfg.training.max_seq_len, runtime_cfg=cfg.runtime)
        if hasattr(model, 'to_empty'):
            model.to_empty(device=device)
        else:
            model = model.to(device)
        if hasattr(model, 'init_weights'):
            model.init_weights()

        # Optional checkpoint load if trainer-format checkpoint exists.
        if cfg.runtime.checkpoint_path and args.checkpoint_step > 0:
            base_dir = cfg.runtime.checkpoint_path
            model_path = os.path.join(base_dir, f"model_{args.checkpoint_step:06d}.pt")
            state = torch.load(model_path, map_location=device)
            target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            target_model.load_state_dict(state, strict=True)
            print0(f"Loaded checkpoint step {args.checkpoint_step} from {base_dir}")

        model.eval()
        from everdream.data.dataloader import tokenizing_weighted_data_loader_bos_bestfit
        val_loader = tokenizing_weighted_data_loader_bos_bestfit(
            tokenizer, cfg.datasets, B=cfg.training.device_batch_size, T=cfg.training.max_seq_len, split='val', seed=cfg.runtime.seed, device=device.type
        )
        modes = [m.strip() for m in args.eval.split(',') if m.strip()]
        results = run_eval(
            model=model,
            tokenizer=tokenizer,
            device=device,
            val_loader=val_loader,
            eval_tokens=args.split_tokens,
            eval_batch_tokens=cfg.training.device_batch_size * cfg.training.max_seq_len,
            eval_modes=modes,
            core_metric_max_per_task=cfg.training.core_metric_max_per_task,
        )
        print0(results)
    finally:
        compute_cleanup(ddp)


if __name__ == "__main__":
    main()
