from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tomllib
import yaml

from .schema import (
    DatasetConfig,
    DenseCustomConfig,
    DenseNanochatConfig,
    EverdreamConfig,
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
    tok_raw = dict(raw.get("tokenizer", {}))
    if "ensure_nanochat_special_tokens" in tok_raw and "ensure_chat_special_tokens" not in tok_raw:
        tok_raw["ensure_chat_special_tokens"] = tok_raw.pop("ensure_nanochat_special_tokens")
    family = raw.get("model", {}).get("family", "dense_custom")
    model_cls = {
        "dense_custom": DenseCustomConfig,
        "dense_nanochat": DenseNanochatConfig,
    }[family]
    cfg = EverdreamConfig(
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        tokenizer=TokenizerConfig(**tok_raw),
        training=TrainingConfig(**raw.get("training", {})),
        model=asdict(model_cls(**raw.get("model", {}))),
        datasets=[DatasetConfig(**d) for d in raw.get("datasets", [])],
    )
    if not cfg.datasets:
        raise ValueError("Config must define at least one dataset")
    return cfg
