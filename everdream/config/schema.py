from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DatasetConfig:
    name: str
    source: str
    split: str = "train"
    weight: float = 1.0
    local_dir: str | None = None
    shard_glob: str = "*.parquet"
    max_shards: int | None = None
    filename_template: str = "shard_{index:05d}.parquet"
    max_shard_index: int | None = None
    num_train_shards: int = -1
    val_shard_index: int | None = None
    num_download_workers: int = 4


@dataclass
class TokenizerConfig:
    source: str = "local"
    path: str | None = None
    vocab_size: int = 32768
    ensure_chat_special_tokens: bool = True
    trainer: str = "rustbpe"
    output_dir: str = ""
    train_split: str = "train"
    train_max_chars: int = 2_000_000_000
    train_doc_cap: int = 10_000
    train_tokenizer_batch_size: int = 128


@dataclass
class RuntimeConfig:
    run_name: str = "dummy"
    device_type: str = ""
    hf_token: str = ""
    wandb_api_key: str = ""
    fp8: bool = False
    fp8_recipe: str = "tensorwise"
    compile: bool = True
    checkpoint_mode: str = "off"
    checkpoint_pattern: str = "CCN"
    no_ckpt_last_n: int = 4
    seed: int = 42
    notebook: bool = False
    mount_drive: bool = False
    drive_path: str = "/content/drive"
    install_gpu_extras: bool = True
    install_moe: bool = False
    install_hybrid: bool = False
    wandb: bool = False
    wandb_project: str = "everdream"
    wandb_mode: str = "online"
    output_dir: str = ""
    checkpoint_path: str = ""


@dataclass
class TrainingConfig:
    max_seq_len: int = 2048
    device_batch_size: int = 16
    total_batch_size: int = 524288
    num_iterations: int = -1
    target_tokens: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 10.5
    eval_every: int = 250
    eval_tokens: int = 80 * 524288
    eval_modes: list[str] = field(default_factory=lambda: ["bpb", "sample"])
    split_tokens: int = 40 * 524288
    core_metric_every: int = -1
    core_metric_max_per_task: int = -1
    sample_every: int = -1
    save_every: int = 1000
    resume_from_step: int = -1
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0
    bench_every: int = -1
    bench_max: int = 200


@dataclass
class DenseNanochatConfig:
    family: Literal["dense_nanochat"] = "dense_nanochat"
    depth: int = 20
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "SSSL"


@dataclass
class DenseCustomConfig:
    family: Literal["dense_custom"] = "dense_custom"
    depth: int = 20
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "L"


@dataclass
class AttentionMoeConfig:
    family: Literal["attention_moe"] = "attention_moe"
    depth: int = 20
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "L"
    moe_num_experts: int = 32
    moe_top_k: int = 2
    moe_inter_dim: int = 512
    moe_z_loss_alpha: float = 1e-3
    moe_jitter_noise: float = 0.1
    moe_loss_weight: float = 1e-2
    dense_first_n: int = 1


@dataclass
class HybridModelConfig:
    family: Literal["hybrid"] = "hybrid"
    depth: int = 16
    d_model: int = 1024
    n_heads: int = 8
    window_pattern: str = "L"
    layer_pattern: str = "DDDADDDADDDADDDA"
    chunk_size: int = 64


ModelConfig = DenseNanochatConfig | DenseCustomConfig | AttentionMoeConfig | HybridModelConfig


@dataclass
class EverdreamConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: dict[str, Any] = field(default_factory=lambda: {"family": "dense_custom"})
    datasets: list[DatasetConfig] = field(default_factory=list)
