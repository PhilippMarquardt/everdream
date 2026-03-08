from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tomllib
import yaml

from .schema import (
    AttentionMoeConfig,
    DatasetConfig,
    DenseCustomConfig,
    DenseNanochatConfig,
    EverdreamConfig,
    HybridModelConfig,
    RuntimeConfig,
    TokenizerConfig,
    TrainingConfig,
)


def _load_mapping(path: Path) -> dict:
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config format: {path}")


def load_config(path: str | Path) -> EverdreamConfig:
    path = Path(path)
    raw = _load_mapping(path)
    family = raw.get("model", {}).get("family", "dense")
    model_cls = {
        "dense": DenseNanochatConfig,
        "dense_nanochat": DenseNanochatConfig,
        "dense_custom": DenseCustomConfig,
        "attention_moe": AttentionMoeConfig,
        "hybrid": HybridModelConfig,
    }[family]
    cfg = EverdreamConfig(
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
        training=TrainingConfig(**raw.get("training", {})),
        model=asdict(model_cls(**raw.get("model", {}))),
        datasets=[DatasetConfig(**d) for d in raw.get("datasets", [])],
    )
    if not cfg.datasets:
        raise ValueError("Config must define at least one dataset")
    return cfg
