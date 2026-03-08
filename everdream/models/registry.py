from __future__ import annotations

from everdream.config.schema import AttentionMoeConfig, DenseCustomConfig, DenseNanochatConfig, HybridModelConfig


def build_model(model_cfg: dict, vocab_size: int, sequence_len: int, runtime_cfg=None):
    family = model_cfg["family"]
    if family in {"dense", "dense_nanochat"}:
        from everdream.models import dense
        cfg = DenseNanochatConfig(**model_cfg)
        d_model = ((cfg.depth * cfg.aspect_ratio + cfg.head_dim - 1) // cfg.head_dim) * cfg.head_dim
        n_heads = d_model // cfg.head_dim
        return dense.GPT(
            dense.GPTConfig(
                sequence_len=sequence_len,
                vocab_size=vocab_size,
                n_layer=cfg.depth,
                n_head=n_heads,
                n_kv_head=n_heads,
                n_embd=d_model,
                window_pattern=cfg.window_pattern,
            )
        )
    if family == "dense_custom":
        from everdream.models import dense_custom
        cfg = DenseCustomConfig(**model_cfg)
        return dense_custom.build_model_from_config(cfg, vocab_size=vocab_size, sequence_len=sequence_len, runtime_cfg=runtime_cfg)
    if family == "attention_moe":
        from everdream.models import attention_moe
        cfg = AttentionMoeConfig(**model_cfg)
        return attention_moe.build_model_from_config(cfg, vocab_size=vocab_size, sequence_len=sequence_len)
    if family == "hybrid":
        from everdream.models import hybrid
        cfg = HybridModelConfig(**model_cfg)
        return hybrid.build_model_from_config(cfg, vocab_size=vocab_size, sequence_len=sequence_len)
    raise ValueError(f"Unknown model family: {family}")
