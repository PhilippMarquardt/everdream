from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from everdream.evaluation.verifiers import RewardSpec


@dataclass
class RLModelConfig:
    name_or_path: str = ""
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"  # "flash_attention_2" on Ampere+ with flash-attn installed
    trust_remote_code: bool = False


@dataclass
class RLDataConfig:
    source: str = "jsonl"  # jsonl | hf
    path: str = ""  # jsonl file path, or HF dataset name when source == "hf"
    paths: list[str] = field(default_factory=list)  # multi-env mixture: jsonl files concatenated + shuffled (overrides path)
    split: str = "train"
    prompt_field: str = "prompt"
    max_samples: int = -1
    # When true, wrap prompts as chat messages and let the tokenizer chat template format them.
    chat: bool = False
    system_prompt: str = ""


@dataclass
class RLEvalConfig:
    suite: str = ""  # path to an eval suite YAML; empty disables eval
    every_steps: int = 0  # run during training every N optimizer steps (0 = only at end)
    at_end: bool = True


@dataclass
class RLTrainConfig:
    output_dir: str = "out/rl"
    run_name: str = "grpo"
    seed: int = 42
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = 10
    max_steps: int = 1000
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    bf16: bool = True
    # GRPO
    num_generations: int = 8
    num_iterations: int = 1  # policy updates per rollout batch (mu)
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.0  # KL coefficient; 0 disables the reference model entirely
    # Generation backend
    use_vllm: bool = False
    vllm_mode: str = "colocate"  # colocate shares the training GPUs; server needs a separate process
    # Logging / saving
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = 3
    wandb: bool = False
    wandb_project: str = "everdream-rl"
    resume_from_checkpoint: bool = False


@dataclass
class RLConfig:
    model: RLModelConfig = field(default_factory=RLModelConfig)
    data: RLDataConfig = field(default_factory=RLDataConfig)
    train: RLTrainConfig = field(default_factory=RLTrainConfig)
    rewards: list[RewardSpec] = field(default_factory=list)
    eval: RLEvalConfig = field(default_factory=RLEvalConfig)


def load_rl_config(path: str | Path) -> RLConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = RLConfig(
        model=RLModelConfig(**raw.get("model", {})),
        data=RLDataConfig(**raw.get("data", {})),
        train=RLTrainConfig(**raw.get("train", {})),
        rewards=[RewardSpec(**r) for r in raw.get("rewards", [])],
        eval=RLEvalConfig(**raw.get("eval", {})),
    )
    if not cfg.model.name_or_path:
        raise ValueError("model.name_or_path is required")
    if not cfg.data.path and not cfg.data.paths:
        raise ValueError("data.path or data.paths is required")
    if not cfg.rewards:
        raise ValueError("At least one reward must be configured")
    return cfg
