from __future__ import annotations

import json
import os

from .config import RLConfig
from .rewards import build_reward_funcs


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prompt_dataset(cfg: RLConfig):
    import random

    from datasets import Dataset, load_dataset

    data = cfg.data
    if data.source == "jsonl":
        rows = []
        for path in (data.paths or [data.path]):
            env_rows = _read_jsonl(path)
            print(f"[data] {path}: {len(env_rows)} prompts")
            rows.extend(env_rows)
        # Environment mixture: every batch samples across envs, never sequential.
        random.Random(cfg.train.seed).shuffle(rows)
        # Normalize to the union of columns (envs carry different verifier
        # columns; schema inference would otherwise drop the missing ones).
        all_keys = sorted(set().union(*(r.keys() for r in rows)))
        ds = Dataset.from_list([{k: r.get(k) for k in all_keys} for r in rows])
    elif data.source == "hf":
        ds = load_dataset(data.path, split=data.split)
    else:
        raise ValueError(f"Unknown data source: {data.source}")

    if data.max_samples > 0:
        ds = ds.select(range(min(data.max_samples, len(ds))))

    def to_prompt(row):
        text = row[data.prompt_field]
        if data.chat:
            messages = []
            if data.system_prompt:
                messages.append({"role": "system", "content": data.system_prompt})
            messages.append({"role": "user", "content": text})
            return {"prompt": messages}
        prefix = f"{data.system_prompt}\n\n" if data.system_prompt else ""
        return {"prompt": prefix + text}

    remove = [data.prompt_field] if data.prompt_field != "prompt" else []
    return ds.map(to_prompt, remove_columns=remove)


def _run_eval_suite(model, tokenizer, suite, header):
    from everdream.evaluation import HFAdapter, flatten_results, print_results, run_suite

    results = run_suite(HFAdapter(model, tokenizer), suite)
    print_results(results, header=header)
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(flatten_results(results))
    except ImportError:
        pass
    return results


def _make_eval_callback(suite, every_steps):
    from transformers import TrainerCallback

    class SuiteEvalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
            if state.global_step % every_steps == 0 and state.is_world_process_zero:
                _run_eval_suite(model, processing_class, suite, header=f"eval @ step {state.global_step}")

    return SuiteEvalCallback()


def train(cfg: RLConfig):
    import torch
    from trl import GRPOConfig, GRPOTrainer

    if cfg.train.wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.train.wandb_project)

    dataset = load_prompt_dataset(cfg)
    reward_funcs, reward_weights = build_reward_funcs(cfg.rewards)

    eval_suite = None
    if cfg.eval.suite:
        from everdream.evaluation import load_eval_suite

        eval_suite = load_eval_suite(cfg.eval.suite)

    grpo_kwargs = dict(
        output_dir=cfg.train.output_dir,
        run_name=cfg.train.run_name,
        seed=cfg.train.seed,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_steps=cfg.train.warmup_steps,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        bf16=cfg.train.bf16,
        num_generations=cfg.train.num_generations,
        num_iterations=cfg.train.num_iterations,
        max_prompt_length=cfg.train.max_prompt_length,
        max_completion_length=cfg.train.max_completion_length,
        temperature=cfg.train.temperature,
        top_p=cfg.train.top_p,
        beta=cfg.train.beta,
        reward_weights=reward_weights,
        use_vllm=cfg.train.use_vllm,
        vllm_mode=cfg.train.vllm_mode,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        report_to=["wandb"] if cfg.train.wandb else [],
        model_init_kwargs={
            "torch_dtype": getattr(torch, cfg.model.torch_dtype),
            "attn_implementation": cfg.model.attn_implementation,
            "trust_remote_code": cfg.model.trust_remote_code,
        },
    )
    # TRL renames/drops GRPOConfig fields across versions (e.g. max_prompt_length
    # gone in 1.8); pass only what the installed version accepts.
    supported = GRPOConfig.__dataclass_fields__
    dropped = sorted(k for k in grpo_kwargs if k not in supported)
    if dropped:
        print(f"[grpo] dropping unsupported GRPOConfig args for this TRL version: {dropped}")
    args = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in supported})

    callbacks = []
    if eval_suite is not None and cfg.eval.every_steps > 0:
        callbacks.append(_make_eval_callback(eval_suite, cfg.eval.every_steps))

    trainer = GRPOTrainer(
        model=cfg.model.name_or_path,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
        callbacks=callbacks or None,
    )
    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint or None)
    trainer.save_model(os.path.join(cfg.train.output_dir, "final"))
    if eval_suite is not None and cfg.eval.at_end and trainer.accelerator.is_main_process:
        _run_eval_suite(trainer.model, trainer.processing_class, eval_suite, header="final eval")
