from __future__ import annotations

import sys
from pathlib import Path

try:
    from nanochat.nanochat.gpt import GPT as NanochatGPT
    from nanochat.nanochat.gpt import GPTConfig as NanochatGPTConfig
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    vendored_nanochat = repo_root / "nanochat"
    if vendored_nanochat.exists():
        sys.path.insert(0, str(vendored_nanochat))
    try:
        from nanochat.gpt import GPT as NanochatGPT
        from nanochat.gpt import GPTConfig as NanochatGPTConfig
    except ModuleNotFoundError:
        from nanochat.nanochat.gpt import GPT as NanochatGPT
        from nanochat.nanochat.gpt import GPTConfig as NanochatGPTConfig

from everdream.config.schema import DenseNanochatConfig


class GPT(NanochatGPT):
    def num_scaling_params(self):
        counts = super().num_scaling_params()
        blocks = counts["transformer_matrices"]
        lm_head = counts["lm_head"]
        total = counts["total"]
        return {
            "wte": counts["wte"],
            "value_embeds": counts["value_embeds"],
            "lm_head": lm_head,
            "blocks": blocks,
            "scalars": counts["scalars"],
            "scaling": blocks + lm_head,
            "active": total,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
    ):
        return super().setup_optimizer(
            unembedding_lr=unembedding_lr,
            embedding_lr=embedding_lr,
            matrix_lr=matrix_lr,
            weight_decay=weight_decay,
            adam_betas=adam_betas,
            scalar_lr=scalar_lr,
        )


def build_model_from_config(cfg: DenseNanochatConfig, vocab_size: int, sequence_len: int, runtime_cfg=None):
    d_model = ((cfg.depth * cfg.aspect_ratio + cfg.head_dim - 1) // cfg.head_dim) * cfg.head_dim
    n_heads = d_model // cfg.head_dim
    n_kv_heads = n_heads if cfg.kv_heads <= 0 else cfg.kv_heads
    return GPT(
        NanochatGPTConfig(
            sequence_len=sequence_len,
            vocab_size=vocab_size,
            n_layer=cfg.depth,
            n_head=n_heads,
            n_kv_head=n_kv_heads,
            n_embd=d_model,
            window_pattern=cfg.window_pattern,
        )
    )
