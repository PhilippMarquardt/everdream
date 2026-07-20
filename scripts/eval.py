"""Standalone evaluation: one suite runner for every phase.

Native everdream checkpoint (pre/mid-training):
    python scripts/eval.py --suite configs/eval_suite_pretrain.yaml \
        --source everdream --config configs/dense_d26.yaml --checkpoint-step 5000

HF model (posttraining, or any baseline):
    python scripts/eval.py --suite configs/eval_suite_json.yaml \
        --source hf --model out/rl_json/final
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from everdream.common import print_banner
from everdream.evaluation import EvalContext, EverdreamAdapter, HFAdapter, load_eval_suite, print_results, run_suite
from everdream.runtime.distributed import compute_cleanup, compute_init


def build_everdream_adapter(args):
    from everdream.config.load import load_config
    from everdream.data.dataloader import tokenizing_weighted_data_loader_bos_bestfit
    from everdream.data.sources import ensure_dataset_ready
    from everdream.models.registry import build_model
    from everdream.runtime.notebook import init_notebook
    from everdream.tokenizer import get_tokenizer

    cfg = load_config(args.config)
    if cfg.runtime.hf_token:
        os.environ["HF_TOKEN"] = cfg.runtime.hf_token
    if cfg.runtime.notebook:
        init_notebook(
            mount_drive=cfg.runtime.mount_drive,
            drive_path=cfg.runtime.drive_path,
            install_gpu_extras=cfg.runtime.install_gpu_extras,
            install_moe=cfg.runtime.install_moe,
            install_hybrid=cfg.runtime.install_hybrid,
        )
    _, ddp_rank, _, _, device = compute_init(cfg.runtime.device_type or "cpu")
    if ddp_rank == 0:
        for spec in cfg.datasets:
            ensure_dataset_ready(spec)

    tokenizer = get_tokenizer(
        tokenizer_dir=cfg.tokenizer.path,
        source=cfg.tokenizer.source,
        ensure_chat_special_tokens=cfg.tokenizer.ensure_chat_special_tokens,
    )
    model = build_model(cfg.model, vocab_size=tokenizer.get_vocab_size(), sequence_len=cfg.training.max_seq_len, runtime_cfg=cfg.runtime)
    model = model.to(device)
    if hasattr(model, "init_weights"):
        model.init_weights()
    if cfg.runtime.checkpoint_path and args.checkpoint_step > 0:
        model_path = os.path.join(cfg.runtime.checkpoint_path, f"model_{args.checkpoint_step:06d}.pt")
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint step {args.checkpoint_step} from {cfg.runtime.checkpoint_path}")
    model.eval()

    def val_loader_factory():
        return tokenizing_weighted_data_loader_bos_bestfit(
            tokenizer, cfg.datasets, B=cfg.training.device_batch_size, T=cfg.training.max_seq_len,
            split="val", seed=cfg.runtime.seed, device=device.type,
        )

    ctx = EvalContext(
        val_loader_factory=val_loader_factory,
        eval_tokens=cfg.training.eval_tokens,
        eval_batch_tokens=cfg.training.device_batch_size * cfg.training.max_seq_len,
    )
    return EverdreamAdapter(model, tokenizer, device), ctx


def main():
    parser = argparse.ArgumentParser(description="Run an evaluation suite against any model")
    parser.add_argument("--suite", required=True, help="Path to eval suite YAML")
    parser.add_argument("--source", choices=["everdream", "hf"], required=True)
    # everdream source
    parser.add_argument("--config", help="Pretrain YAML config (everdream source)")
    parser.add_argument("--checkpoint-step", type=int, default=-1)
    # hf source
    parser.add_argument("--model", help="HF model name or local path (hf source)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--output", help="Optional path to write results JSON")
    args = parser.parse_args()

    print_banner()
    suite = load_eval_suite(args.suite)
    try:
        if args.source == "everdream":
            assert args.config, "--config is required with --source everdream"
            adapter, ctx = build_everdream_adapter(args)
        else:
            assert args.model, "--model is required with --source hf"
            adapter, ctx = HFAdapter.from_pretrained(args.model, torch_dtype=args.dtype, attn_implementation=args.attn), None

        results = run_suite(adapter, suite, ctx)
        print_results(results, header=suite.name)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Wrote results to {args.output}")
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
