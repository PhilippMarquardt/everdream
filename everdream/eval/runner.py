from __future__ import annotations

import csv
from contextlib import contextmanager
import json
import os
import random
import shutil
import tempfile
import time
import zipfile
from collections.abc import Iterable

import torch
import yaml

from everdream.common import download_file_with_lock, get_base_dir, print0
from everdream.eval.metrics import evaluate_bpb, evaluate_task
from everdream.tokenizer import get_token_bytes

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def _place_eval_bundle(file_path):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


@contextmanager
def disable_fp8(model):
    import torch.nn as nn

    class EvalLinear(nn.Linear):
        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight.to(dtype=x.dtype), None if self.bias is None else self.bias.to(dtype=x.dtype))

    fp8_locations = []
    for name, module in model.named_modules():
        if "Float8" not in type(module).__name__:
            continue
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name
        fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield
        return

    for parent, attr_name, fp8_module in fp8_locations:
        linear = EvalLinear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)


@torch.no_grad()
def evaluate_core(model, tokenizer, device, max_per_task=-1):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=_place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with open(eval_meta_data, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end="")
        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered:.4f} | time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}


@torch.no_grad()
def generate_samples(model, tokenizer, prompts: list[str], max_tokens: int = 128):
    if not hasattr(model, "generate"):
        return None
    was_training = model.training
    model.eval()
    outputs = []
    try:
        with disable_fp8(model):
            bos_text = tokenizer.decode([tokenizer.get_bos_token_id()])
            for prompt in prompts:
                prompt_tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
                generated = []
                stream = model.generate(prompt_tokens, max_tokens=max_tokens, temperature=0.8, top_k=50)
                for token in stream:
                    generated.append(token)
                decoded = tokenizer.decode(prompt_tokens + generated)
                if bos_text and decoded.startswith(bos_text):
                    decoded = decoded[len(bos_text):]
                outputs.append(decoded)
    finally:
        if was_training:
            model.train()
    return outputs


def run_eval(
    model,
    tokenizer,
    device,
    val_loader: Iterable,
    eval_tokens: int,
    eval_batch_tokens: int,
    eval_modes: list[str],
    core_metric_max_per_task: int = -1,
):
    results: dict[str, object] = {}
    modes = set(eval_modes)

    if "bpb" in modes:
        token_bytes = get_token_bytes(device=device, tokenizer=tokenizer)
        steps = max(1, eval_tokens // eval_batch_tokens)
        with disable_fp8(model):
            results["bpb"] = evaluate_bpb(model, val_loader, steps, token_bytes)

    if "core" in modes:
        with disable_fp8(model):
            results["core"] = evaluate_core(model, tokenizer, device, max_per_task=core_metric_max_per_task)

    if "sample" in modes:
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If 5*x + 3 = 13, then x is",
            "def fibonacci(n):",
        ]
        samples = generate_samples(model, tokenizer, prompts)
        if samples is not None:
            results["samples"] = samples

    return results
