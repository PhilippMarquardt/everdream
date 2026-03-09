from __future__ import annotations

from everdream.config.schema import DenseCustomConfig


def build_model(model_cfg: dict, vocab_size: int, sequence_len: int, runtime_cfg=None):
    family = model_cfg["family"]
    if family == "dense_custom":
        from everdream.models import dense_custom
        cfg = DenseCustomConfig(**model_cfg)
        return dense_custom.build_model_from_config(cfg, vocab_size=vocab_size, sequence_len=sequence_len, runtime_cfg=runtime_cfg)
    raise ValueError(f"Unknown model family: {family}")
