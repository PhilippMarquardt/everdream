import argparse
import os
from collections import Counter
from collections import defaultdict

from everdream.common import print_banner
from everdream.config.load import load_config
from everdream.data.dataloader import tokenizing_weighted_data_loader_bos_bestfit
from everdream.data.sources import document_batches
from everdream.data.sources import ensure_dataset_ready
from everdream.tokenizer import get_tokenizer


def reconstruct_row(x_row, y_row):
    return x_row.tolist() + [int(y_row[-1])]


def inspect_document_lengths(cfg, tokenizer, split: str, per_source_docs: int):
    bos_id = tokenizer.get_bos_token_id()
    print("\n=== raw document token lengths ===")
    for spec in cfg.datasets:
        docs = []
        iterator = document_batches(spec, split=split)
        while len(docs) < per_source_docs:
            try:
                docs.extend(next(iterator))
            except StopIteration:
                break
        docs = docs[:per_source_docs]
        if not docs:
            print(f"{spec.name}: no docs")
            continue
        token_lists = tokenizer.encode(docs, prepend=bos_id)
        lengths = [len(tokens) for tokens in token_lists]
        over_ctx = sum(1 for n in lengths if n > cfg.training.max_seq_len + 1)
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        print(
            f"{spec.name}: docs={len(lengths)} avg={avg_len:.1f} min={min_len} "
            f"max={max_len} over_ctx={over_ctx}/{len(lengths)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Inspect packed training rows from the current loader")
    parser.add_argument("--config", required=True, help="Path to YAML/TOML config")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--batches", type=int, default=2, help="How many batches to inspect")
    parser.add_argument("--rows", type=int, default=4, help="How many rows per batch to print")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Loader output device")
    parser.add_argument("--doc-stats", type=int, default=32, help="Number of raw docs per source for token-length stats")
    args = parser.parse_args()

    print_banner()
    cfg = load_config(args.config)
    if cfg.runtime.hf_token:
        os.environ["HF_TOKEN"] = cfg.runtime.hf_token

    for spec in cfg.datasets:
        ensure_dataset_ready(spec)

    tokenizer = get_tokenizer(
        tokenizer_dir=cfg.tokenizer.path,
        source=cfg.tokenizer.source,
        ensure_chat_special_tokens=cfg.tokenizer.ensure_chat_special_tokens,
    )
    inspect_document_lengths(cfg, tokenizer, split=args.split, per_source_docs=args.doc_stats)
    bos_id = tokenizer.get_bos_token_id()

    loader = tokenizing_weighted_data_loader_bos_bestfit(
        tokenizer,
        cfg.datasets,
        B=cfg.training.device_batch_size,
        T=cfg.training.max_seq_len,
        split=args.split,
        seed=cfg.runtime.seed,
        device=args.device,
    )

    total_sources = Counter()
    max_rows = min(args.rows, cfg.training.device_batch_size)

    for batch_idx in range(args.batches):
        x, y, state = next(loader)
        batch_sources = state.get("sources", [])
        total_sources.update(batch_sources)
        print(f"\n=== batch {batch_idx + 1} / {args.batches} | epoch={state.get('epoch')} draws={state.get('draws')} ===")
        print(f"row sources: {Counter(batch_sources)}")
        for row_idx in range(max_rows):
            source = batch_sources[row_idx] if row_idx < len(batch_sources) else "unknown"
            row = reconstruct_row(x[row_idx], y[row_idx])
            bos_count = sum(1 for tok in row if tok == bos_id)
            decoded = tokenizer.decode(row)
            print(f"\n--- row {row_idx} | source={source} | len={len(row)} | bos_count={bos_count} ---")
            print(decoded[:2000])

    print("\n=== aggregate row source counts ===")
    for source, count in total_sources.most_common():
        print(f"{source}: {count}")


if __name__ == "__main__":
    main()
