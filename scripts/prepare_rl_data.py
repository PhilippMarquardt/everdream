"""Prepare RL environments (dataset + verifier columns) for GRPO training + eval.

Each environment writes train/eval JSONL with the columns our verifiers consume:
prompt, ground_truth (+ lower_limit/upper_limit where the source provides ranges).

    python scripts/prepare_rl_data.py                    # all envs
    python scripts/prepare_rl_data.py --task gsm8k
    python scripts/prepare_rl_data.py --task medcalc
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_GSM8K_ANSWER_RE = re.compile(r"####\s*(.+)")


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")


def prepare_gsm8k(args):
    from datasets import load_dataset

    def convert(dataset, cap):
        rows = []
        for row in dataset:
            m = _GSM8K_ANSWER_RE.search(row["answer"])
            if not m:
                continue
            rows.append({"prompt": row["question"], "ground_truth": m.group(1).strip().replace(",", "")})
            if cap > 0 and len(rows) >= cap:
                break
        return rows

    out = Path(args.out_dir) / "gsm8k"
    for split, name, cap in (("train", "train.jsonl", args.max_samples), ("test", "eval.jsonl", args.max_eval_samples)):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        write_jsonl(convert(ds, cap), out / name)


def prepare_medcalc(args):
    from datasets import load_dataset

    def convert(dataset, cap):
        rows = []
        for row in dataset:
            # Numeric outputs only; date/text calculators need a different verifier.
            if row["Output Type"] not in ("decimal", "integer"):
                continue
            prompt = f"Patient note:\n{row['Patient Note']}\n\nQuestion: {row['Question']}"
            rows.append({
                "prompt": prompt,
                "ground_truth": str(row["Ground Truth Answer"]),
                "lower_limit": str(row["Lower Limit"]),
                "upper_limit": str(row["Upper Limit"]),
            })
            if cap > 0 and len(rows) >= cap:
                break
        return rows

    out = Path(args.out_dir) / "medcalc"
    for split, name, cap in (("train", "train.jsonl", args.max_samples), ("test", "eval.jsonl", args.max_eval_samples)):
        ds = load_dataset("ncbi/MedCalc-Bench-v1.2", split=split)
        write_jsonl(convert(ds, cap), out / name)


def prepare_banking77(args):
    from datasets import load_dataset

    # mteb mirror: the original PolyAI repo is a legacy script dataset (unsupported).
    ds_train = load_dataset("mteb/banking77", split="train")
    label_names = sorted(set(ds_train["label_text"]))
    label_list = "\n".join(f"- {name}" for name in label_names)

    def convert(dataset, cap):
        rows = []
        for row in dataset:
            prompt = (
                "Classify the customer message into exactly one banking intent.\n\n"
                f"Possible intents:\n{label_list}\n\n"
                f"Customer message: {row['text']}\n\n"
                "Reply with the single best-matching intent from the list."
            )
            rows.append({"prompt": prompt, "ground_truth": row["label_text"]})
            if cap > 0 and len(rows) >= cap:
                break
        return rows

    out = Path(args.out_dir) / "banking77"
    for split, name, cap in (("train", "train.jsonl", args.max_samples), ("test", "eval.jsonl", args.max_eval_samples)):
        ds = load_dataset("mteb/banking77", split=split)
        write_jsonl(convert(ds, cap), out / name)


TASKS = {"gsm8k": prepare_gsm8k, "medcalc": prepare_medcalc, "banking77": prepare_banking77}


def main():
    parser = argparse.ArgumentParser(description="Prepare RL environment data")
    parser.add_argument("--task", default="all", choices=["all", *TASKS])
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--max-samples", type=int, default=-1, help="Cap train rows per env")
    parser.add_argument("--max-eval-samples", type=int, default=500, help="Cap eval rows per env (from test split)")
    args = parser.parse_args()

    for name in TASKS if args.task == "all" else [args.task]:
        print(f"== {name} ==")
        TASKS[name](args)


if __name__ == "__main__":
    main()
