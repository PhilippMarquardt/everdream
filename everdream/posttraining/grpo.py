from __future__ import annotations

import json
import os

from .config import RLConfig
from .rewards import build_reward_funcs


def load_prompt_dataset(cfg: RLConfig):
    from datasets import Dataset, load_dataset

    data = cfg.data
    if data.source == "jsonl":
        rows = []
        with open(data.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        ds = Dataset.from_list(rows)
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


def train(cfg: RLConfig):
    import torch
    from trl import GRPOConfig, GRPOTrainer

    if cfg.train.wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.train.wandb_project)

    dataset = load_prompt_dataset(cfg)
    reward_funcs, reward_weights = build_reward_funcs(cfg.rewards)

    args = GRPOConfig(
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

    trainer = GRPOTrainer(
        model=cfg.model.name_or_path,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint or None)
    trainer.save_model(os.path.join(cfg.train.output_dir, "final"))
