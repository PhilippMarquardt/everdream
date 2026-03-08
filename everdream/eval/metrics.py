from __future__ import annotations

import math
import random
from typing import Iterable

import torch
import torch.distributed as dist
from jinja2 import Template


def _model_device(model) -> torch.device:
    if hasattr(model, "get_device"):
        return model.get_device()
    return next(model.parameters()).device


def _model_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple):
        return out[0]
    raise TypeError(f"Unsupported model output type: {type(out)}")


def _loss2d_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    losses = torch.nn.functional.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        ignore_index=-1,
        reduction="none",
    )
    return losses.view(B, T)


@torch.no_grad()
def evaluate_bpb(model, batches: Iterable, steps: int, token_bytes: torch.Tensor) -> float:
    device = _model_device(model)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        batch = next(batch_iter)
        x, y = batch[0], batch[1]
        logits = _model_logits(model, x)
        loss2d = _loss2d_from_logits(logits, y).view(-1)
        y = y.view(-1)
        if (y.int() < 0).any():
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y, dtype=token_bytes.dtype))
            total_nats += (loss2d * (num_bytes > 0)).sum()
            total_bytes += num_bytes.sum()
        else:
            num_bytes = token_bytes[y]
            total_nats += (loss2d * (num_bytes > 0)).sum()
            total_bytes += num_bytes.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_bytes_i = total_bytes.item()
    if total_bytes_i == 0:
        return float("inf")
    return total_nats.item() / (math.log(2) * total_bytes_i)


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    template = Template(
        """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    )
    fewshot_examples = fewshot_examples or []
    ctx = {"fewshot_examples": fewshot_examples, "continuation_delimiter": continuation_delimiter, "item": item}
    return [template.render(choice=choice, **ctx) for choice in item["choices"]]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    template = Template(
        """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    )
    fewshot_examples = fewshot_examples or []
    ctx = {"fewshot_examples": fewshot_examples, "continuation_delimiter": continuation_delimiter, "item": item}
    return [template.render(context=context_option, **ctx) for context_option in item["context_options"]]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    template = Template(
        """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    )
    fewshot_examples = fewshot_examples or []
    ctx = {"fewshot_examples": fewshot_examples, "continuation_delimiter": continuation_delimiter, "item": item}
    prompt_without = template.render(include_continuation=False, **ctx).strip()
    prompt_with = template.render(include_continuation=True, **ctx)
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    min_len = min(len(seq) for seq in token_sequences)
    indices = {"left": range(min_len), "right": range(-1, -min_len - 1, -1)}[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert tokens_without == tokens_with[:start_idx]
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    outputs = _model_logits(model, input_ids)
    B, T, V = outputs.shape
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = torch.nn.functional.cross_entropy(
        outputs.view(B * T, V),
        target_ids.view(B * T),
        reduction="none",
    ).view(B, T)
    losses[:, -1] = float("nan")
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    max_seq_len = getattr(getattr(model, "config", None), "sequence_len", None)
    if max_seq_len is None and hasattr(model, "max_seq_len"):
        max_seq_len = model.max_seq_len
    if max_seq_len is not None:
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_seq_len:
                num_to_crop = len(t) - max_seq_len
                new_tokens.append(t[-max_seq_len:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, input_ids)

    if task_type == "language_modeling":
        si = start_idxs[0]
        ei = end_idxs[0]
        predicted_tokens = predictions[0, si - 1 : ei - 1]
        actual_tokens = input_ids[0, si:ei]
        return torch.all(predicted_tokens == actual_tokens).item()
    mean_losses = [losses[i, si - 1 : ei - 1].mean().item() for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
    pred_idx = mean_losses.index(min(mean_losses))
    return pred_idx == item["gold"]


def evaluate_task(model, tokenizer, data, device, task_meta):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx in range(rank, len(data), world_size):
        correct[idx] = float(evaluate_example(idx, model, tokenizer, data, device, task_meta))
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()

