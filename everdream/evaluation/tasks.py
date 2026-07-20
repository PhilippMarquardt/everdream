"""Eval task registry.

Each task is fn(adapter, spec, ctx) -> dict of results. Generation-based tasks
(generation, sample) work with any adapter; loss-based tasks (bpb, core) need
the native everdream model and a val loader supplied via EvalContext, so they
are typically used during pre/mid-training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .adapters import ModelAdapter
from .config import EvalTaskSpec
from .verifiers import build_verifiers


@dataclass
class EvalContext:
    """Phase-supplied resources for loss-based tasks."""
    val_loader_factory: Callable | None = None
    eval_tokens: int = 0
    eval_batch_tokens: int = 0


def load_eval_rows(spec: EvalTaskSpec) -> list[dict]:
    data = spec.data
    if data is None:
        raise ValueError(f"Task '{spec.name}' ({spec.type}) requires a data section")
    if data.source == "jsonl":
        rows = []
        with open(data.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif data.source == "hf":
        from datasets import load_dataset

        rows = [dict(r) for r in load_dataset(data.path, split=data.split)]
    else:
        raise ValueError(f"Unknown eval data source: {data.source}")
    if data.max_samples > 0:
        rows = rows[: data.max_samples]
    if not rows:
        raise ValueError(f"Task '{spec.name}': no rows loaded from {data.path}")
    return rows


def build_prompts(rows: list[dict], spec: EvalTaskSpec) -> list:
    data = spec.data
    prompts = []
    for row in rows:
        text = row[data.prompt_field]
        if data.chat:
            messages = []
            if data.system_prompt:
                messages.append({"role": "system", "content": data.system_prompt})
            messages.append({"role": "user", "content": text})
            prompts.append(messages)
        else:
            prefix = f"{data.system_prompt}\n\n" if data.system_prompt else ""
            prompts.append(prefix + text)
    return prompts


def task_generation(adapter: ModelAdapter, spec: EvalTaskSpec, ctx: EvalContext) -> dict:
    if not spec.metrics:
        raise ValueError(f"Generation task '{spec.name}' needs at least one metric")
    rows = load_eval_rows(spec)
    prompts = build_prompts(rows, spec)
    completions = adapter.generate(prompts, spec.gen)

    # Per-sample extra columns (schema, ground_truth, ...) go to verifiers as kwargs.
    extra_cols = {}
    for key in rows[0]:
        if key != spec.data.prompt_field:
            extra_cols[key] = [row.get(key) for row in rows]

    funcs, weights = build_verifiers(spec.metrics)
    metrics, score = {}, 0.0
    for fn, weight in zip(funcs, weights):
        scores = [s for s in fn(prompts, completions, **extra_cols) if s is not None]
        mean = sum(scores) / len(scores) if scores else 0.0
        metrics[fn.__name__] = mean
        score += weight * mean

    num_samples = spec.params.get("log_samples", 3)
    samples = [
        {"prompt": rows[i][spec.data.prompt_field], "completion": completions[i]}
        for i in range(min(num_samples, len(completions)))
    ]
    return {"score": score, "metrics": metrics, "n": len(completions), "samples": samples}


_DEFAULT_SAMPLE_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If 5*x + 3 = 13, then x is",
    "def fibonacci(n):",
]


def task_sample(adapter: ModelAdapter, spec: EvalTaskSpec, ctx: EvalContext) -> dict:
    prompts = spec.params.get("prompts", _DEFAULT_SAMPLE_PROMPTS)
    completions = adapter.generate(prompts, spec.gen)
    return {"samples": [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]}


def task_bpb(adapter: ModelAdapter, spec: EvalTaskSpec, ctx: EvalContext) -> dict:
    if adapter.lm_model is None or ctx.val_loader_factory is None:
        raise ValueError(f"Task '{spec.name}' (bpb) needs a native model and a val loader (pre/mid-training context)")
    from everdream.eval import disable_fp8
    from everdream.eval.metrics import evaluate_bpb
    from everdream.tokenizer import get_token_bytes

    eval_tokens = spec.params.get("eval_tokens", ctx.eval_tokens)
    steps = max(1, eval_tokens // max(1, ctx.eval_batch_tokens))
    token_bytes = get_token_bytes(device=adapter.device, tokenizer=adapter.tokenizer)
    with disable_fp8(adapter.lm_model):
        bpb = evaluate_bpb(adapter.lm_model, ctx.val_loader_factory(), steps, token_bytes)
    return {"bpb": bpb}


def task_core(adapter: ModelAdapter, spec: EvalTaskSpec, ctx: EvalContext) -> dict:
    if adapter.lm_model is None:
        raise ValueError(f"Task '{spec.name}' (core) needs a native everdream model")
    from everdream.eval import disable_fp8
    from everdream.eval.runner import evaluate_core

    with disable_fp8(adapter.lm_model):
        return evaluate_core(
            adapter.lm_model, adapter.tokenizer, adapter.device,
            max_per_task=spec.params.get("max_per_task", -1),
        )


TASK_REGISTRY = {
    "generation": task_generation,
    "sample": task_sample,
    "bpb": task_bpb,
    "core": task_core,
}
