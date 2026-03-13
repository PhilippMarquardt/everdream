from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

from everdream.config.load import load_config
from everdream.common import print_banner
from everdream.data.sources import document_batches
from everdream.data.sources import ensure_dataset_ready
from everdream.data.tokenizer import HuggingFaceTokenizer, RustBPETokenizer, save_token_bytes
from everdream.runtime.notebook import init_notebook
from everdream.runtime.distributed import get_dist_info, print0


def build_text_iterator(cfg):
    max_chars = cfg.tokenizer.train_max_chars
    doc_cap = cfg.tokenizer.train_doc_cap
    seen_chars = 0
    batch_size = cfg.tokenizer.train_tokenizer_batch_size
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    rng = random.Random(cfg.runtime.seed + (ddp_rank if ddp else 0))
    iterators = [
        document_batches(spec, split=cfg.tokenizer.train_split, start=ddp_rank if ddp else 0, step=ddp_world_size if ddp else 1)
        for spec in cfg.datasets
    ]
    pending_docs = [[] for _ in cfg.datasets]
    weights = [float(spec.weight) for spec in cfg.datasets]
    consumed_chars = [0 for _ in cfg.datasets]
    buffered_chars = [0 for _ in cfg.datasets]
    doc_buffer: list[tuple[int, str]] = []

    def next_doc_batch(dataset_idx: int):
        docs = pending_docs[dataset_idx]
        spec = cfg.datasets[dataset_idx]
        while len(docs) < batch_size:
            try:
                docs.extend(next(iterators[dataset_idx]))
            except StopIteration:
                iterators[dataset_idx] = document_batches(
                    spec,
                    split=cfg.tokenizer.train_split,
                    start=ddp_rank if ddp else 0,
                    step=ddp_world_size if ddp else 1,
                )
                docs.extend(next(iterators[dataset_idx]))
        pending_docs[dataset_idx] = docs[batch_size:]
        return docs[:batch_size]

    def refill_buffer():
        if not doc_buffer:
            dataset_idx = rng.choices(range(len(cfg.datasets)), weights=weights, k=1)[0]
        else:
            scheduled_chars = [consumed_chars[i] + buffered_chars[i] for i in range(len(cfg.datasets))]
            dataset_idx = min(
                range(len(cfg.datasets)),
                key=lambda i: scheduled_chars[i] / max(weights[i], 1e-12),
            )
        for doc in next_doc_batch(dataset_idx):
            text = doc[:doc_cap] if len(doc) > doc_cap else doc
            doc_buffer.append((dataset_idx, text))
            buffered_chars[dataset_idx] += len(text)

    while seen_chars < max_chars:
        if not doc_buffer:
            refill_buffer()
        dataset_idx, text = doc_buffer.pop(0)
        buffered_chars[dataset_idx] -= len(text)
        consumed_chars[dataset_idx] += len(text)
        seen_chars += len(text)
        yield text


def train_tokenizer(cfg):
    for spec in cfg.datasets:
        ensure_dataset_ready(spec)

    output_dir = cfg.tokenizer.output_dir or cfg.tokenizer.path
    if not output_dir:
        raise ValueError("tokenizer.output_dir or tokenizer.path must be set for tokenizer training")
    output_dir = str(Path(output_dir))

    print0(f"Tokenizer trainer: {cfg.tokenizer.trainer}")
    print0(f"Tokenizer vocab_size: {cfg.tokenizer.vocab_size:,}")
    print0(f"Tokenizer train_split: {cfg.tokenizer.train_split}")
    print0(f"Tokenizer train_max_chars: {cfg.tokenizer.train_max_chars:,}")
    print0(f"Tokenizer train_doc_cap: {cfg.tokenizer.train_doc_cap:,}")
    print0(f"Tokenizer output_dir: {output_dir}")

    text_iter = build_text_iterator(cfg)
    trainer_name = cfg.tokenizer.trainer.lower()
    if trainer_name == "rustbpe":
        tokenizer = RustBPETokenizer.train_from_iterator(text_iter, cfg.tokenizer.vocab_size)
    elif trainer_name in {"hf", "hf_bpe", "huggingface"}:
        tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iter, cfg.tokenizer.vocab_size)
    else:
        raise ValueError(f"Unsupported tokenizer trainer: {cfg.tokenizer.trainer}")

    tokenizer.save(output_dir)
    token_bytes_path = save_token_bytes(tokenizer, output_dir)
    print0(f"Saved token_bytes to {token_bytes_path}")


def main():
    parser = argparse.ArgumentParser(description="Train an everdream tokenizer from the configured dataset mix")
    parser.add_argument("--config", required=True)
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
    train_tokenizer(cfg)


if __name__ == "__main__":
    main()
