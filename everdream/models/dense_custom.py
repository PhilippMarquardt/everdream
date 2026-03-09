from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils

from everdream.common import COMPUTE_DTYPE, get_dist_info
from everdream.config.schema import DenseCustomConfig
from everdream.flash_attention import flash_attn
from everdream.optim import DistMuonAdamW, MuonAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 26
    n_head: int = 13
    n_embd: int = 1664
    window_pattern: str = "L"
    checkpoint_mode: str = "off"
    checkpoint_pattern: str = "CCN"
    no_ckpt_last_n: int = 4


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def compute_window_sizes(config: GPTConfig):
    pattern = config.window_pattern.upper()
    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
    sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
    sizes[-1] = (long_window, 0)
    return sizes


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, self.n_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_heads, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_heads, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_heads, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        attn_dtype = q.dtype
        k = k.to(attn_dtype)
        v = v.to(attn_dtype)
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate_up = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=False)
        self.down = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.window_sizes = compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        kv_dim = config.n_head * (config.n_embd // config.n_head)
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(config.vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        cos, sin = self._precompute_rope(config.sequence_len * 10, config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rope(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    @torch.no_grad()
    def init_weights(self):
        # Recompute RoPE buffers on current device (required after to_empty)
        device = self.wte.weight.device
        cos, sin = self._precompute_rope(self.config.sequence_len * 10, self.config.n_embd // self.config.n_head)
        self.cos = cos.to(device)
        self.sin = sin.to(device)
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3 ** 0.5 * self.config.n_embd ** -0.5
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.gate_up.weight, -s, s)
            nn.init.zeros_(block.mlp.down.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
        for block in self.blocks:
            if block.attn.ve_gate is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)
        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def estimate_flops(self):
        D, H, d, t = self.config.n_embd, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        dense_block_params = 16 * D * D
        ve_gate_params = sum(32 * H for i in range(self.config.n_layer) if has_ve(i, self.config.n_layer))
        lm_head_params = D * self.config.vocab_size
        total_active = self.config.n_layer * dense_block_params + ve_gate_params + lm_head_params
        matmul_flops = 6 * total_active
        attn_flops = 0
        for ws in self.window_sizes:
            eff_t = t if ws[0] <= 0 else min(ws[0], t)
            attn_flops += 12 * H * d * eff_t
        return matmul_flops + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        block_params = sum(p.numel() for p in self.blocks.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + block_params + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "blocks": block_params,
            "scalars": scalars,
            "active": total,
            "total": total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.3, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        ddp, *_ = get_dist_info()
        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        embedding_lr *= dmodel_lr_scale
        unembedding_lr *= dmodel_lr_scale
        scalar_lr *= dmodel_lr_scale
        matrix_params = list(self.blocks.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind="muon", params=group_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))
        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def should_checkpoint_layer(self, layer_idx):
        if not self.training:
            return False
        mode = self.config.checkpoint_mode
        if mode == "off":
            return False
        if mode == "full":
            return True
        if mode == "patterned":
            if self.config.no_ckpt_last_n > 0 and layer_idx >= self.config.n_layer - self.config.no_ckpt_last_n:
                return False
            pattern = self.config.checkpoint_pattern.upper()
            if not pattern or any(c not in "CN" for c in pattern):
                raise ValueError(f"Invalid checkpoint pattern: {self.config.checkpoint_pattern}")
            return pattern[layer_idx % len(pattern)] == "C"
        raise ValueError(f"Invalid checkpoint mode: {mode}")

    def forward(self, idx, targets=None):
        _, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.wte(idx))
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if self.should_checkpoint_layer(i):
                def block_fn(x_in, _block=block, _ve=ve, _ws=self.window_sizes[i]):
                    return _block(x_in, _ve, cos_sin, _ws)
                x = checkpoint_utils.checkpoint(block_fn, x, use_reentrant=False)
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(idx, list)
        device = self.wte.weight.device
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([idx], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            with torch.autocast(device_type=device.type, dtype=COMPUTE_DTYPE):
                logits, _ = self(ids[:, -self.config.sequence_len :])
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")
                next_id = torch.multinomial(F.softmax(logits, -1), 1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            yield next_id.item()


def build_model_from_config(cfg: DenseCustomConfig, vocab_size: int, sequence_len: int, runtime_cfg=None):
    d_model = ((cfg.depth * cfg.aspect_ratio + cfg.head_dim - 1) // cfg.head_dim) * cfg.head_dim
    n_heads = d_model // cfg.head_dim
    return GPT(
        GPTConfig(
            sequence_len=sequence_len,
            vocab_size=vocab_size,
            n_layer=cfg.depth,
            n_head=n_heads,
            n_embd=d_model,
            window_pattern=cfg.window_pattern,
            checkpoint_mode=getattr(runtime_cfg, "checkpoint_mode", "off"),
            checkpoint_pattern=getattr(runtime_cfg, "checkpoint_pattern", "CCN"),
            no_ckpt_last_n=getattr(runtime_cfg, "no_ckpt_last_n", 4),
        )
    )
