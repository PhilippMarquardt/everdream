from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from everdream.common import COMPUTE_DTYPE, get_dist_info, print0
from everdream.config.schema import AttentionMoeConfig
from everdream.models.dense import GPTConfig, Linear, apply_rotary_emb, has_ve, norm
from everdream.flash_attention import flash_attn
from everdream.optim import DistMuonAdamW, MuonAdamW

try:
    from megablocks.layers.arguments import Arguments as MBArgs
    from megablocks.layers.dmoe import dMoE
    from megablocks.layers import moe as mb_moe
    from megablocks.layers import router as mb_router
except ImportError:
    MBArgs = None
    dMoE = None
    mb_moe = None
    mb_router = None


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class SharedExpertMLP(nn.Module):
    def __init__(self, n_embd: int, inter_dim: int):
        super().__init__()
        self.gate_up = Linear(n_embd, 2 * inter_dim, bias=False)
        self.down = Linear(inter_dim, n_embd, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def _make_mb_args(n_embd: int, cfg: AttentionMoeConfig, num_moe_layers: int):
    if MBArgs is None:
        raise RuntimeError("MegaBlocks is not installed. Install everdream[moe] or run notebook init with install_moe=True.")
    return MBArgs(
        hidden_size=n_embd,
        ffn_hidden_size=cfg.moe_inter_dim,
        moe_num_experts=cfg.moe_num_experts,
        moe_top_k=cfg.moe_top_k,
        moe_capacity_factor=1,
        moe_loss_weight=cfg.moe_loss_weight,
        moe_jitter_eps=cfg.moe_jitter_noise,
        moe_zloss_weight=cfg.moe_z_loss_alpha,
        activation_fn=F.silu,
        bias=False,
        return_bias=False,
        mlp_impl='grouped',
        bf16=True,
        fp16=False,
        num_layers=num_moe_layers,
        init_method=partial(torch.nn.init.normal_, mean=0.0, std=0.02),
    )


class MegaBlocksMoeBlock(nn.Module):
    def __init__(self, n_embd: int, cfg: AttentionMoeConfig, num_moe_layers: int):
        super().__init__()
        self.n_embd = n_embd
        self.cfg = cfg
        self.num_moe_layers = num_moe_layers
        self.dmoe = dMoE(_make_mb_args(n_embd, cfg, num_moe_layers))
        self.shared_expert = SharedExpertMLP(n_embd, cfg.moe_inter_dim)
        self.shared_gate = Linear(n_embd, 1, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        routed = self.dmoe(x)
        x_flat = x.view(-1, D)
        shared = torch.sigmoid(self.shared_gate(x_flat)) * self.shared_expert(x_flat)
        return routed + shared.view(B, T, D)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        attn_dtype = q.dtype
        y = flash_attn.flash_attn_func(q, k.to(attn_dtype), v.to(attn_dtype), causal=True, window_size=window_size)
        return self.c_proj(y.contiguous().view(B, T, -1))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int, mlp: nn.Module):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = mlp

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class AttentionMoeModel(nn.Module):
    def __init__(self, config: GPTConfig, moe_cfg: AttentionMoeConfig):
        super().__init__()
        self.config = config
        self.moe_cfg = moe_cfg
        self.window_sizes = self._compute_window_sizes(config)
        num_moe_layers = max(config.n_layer - moe_cfg.dense_first_n, 1)
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'h': nn.ModuleList([
                Block(config, i, MLP(config.n_embd) if i < moe_cfg.dense_first_n else MegaBlocksMoeBlock(config.n_embd, moe_cfg, num_moe_layers))
                for i in range(config.n_layer)
            ]),
        })
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(config.vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {'L': (long_window, 0), 'S': (short_window, 0)}
        sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_window, 0)
        return sizes

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().to(COMPUTE_DTYPE), freqs.sin().to(COMPUTE_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if isinstance(block.mlp, MLP):
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            else:
                torch.nn.init.zeros_(block.mlp.shared_gate.weight)
                torch.nn.init.uniform_(block.mlp.shared_expert.gate_up.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.shared_expert.down.weight)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

    def estimate_flops(self):
        total_active = self.num_scaling_params()['active']
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * min(w[0], t) for w in self.window_sizes)
        return 6 * total_active + attn_flops

    def num_scaling_params(self):
        total = sum(p.numel() for p in self.parameters())
        total_mlp = sum(sum(p.numel() for p in block.mlp.parameters()) for block in self.transformer.h)
        active_mlp = 0
        for block in self.transformer.h:
            if isinstance(block.mlp, MLP):
                active_mlp += sum(p.numel() for p in block.mlp.parameters())
            else:
                active_mlp += (self.moe_cfg.moe_top_k + 1) * 3 * self.config.n_embd * self.moe_cfg.moe_inter_dim
        return {'total': total, 'active': total - total_mlp + active_mlp}

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, *_ = get_dist_info()
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ?1/v({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))
        factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = factory(param_groups)
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        return optimizer

    def forward(self, idx, targets=None):
        T = idx.size(1)
        if self.training:
            mb_moe.clear_load_balancing_loss()
            if hasattr(mb_router, 'clear_router_zloss'):
                mb_router.clear_router_zloss()
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 20 * torch.tanh(logits / 20)
        if targets is None:
            return logits, None, None, None, None
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        if self.training:
            args = _make_mb_args(self.config.n_embd, self.moe_cfg, max(self.config.n_layer - self.moe_cfg.dense_first_n, 1))
            lb_loss = mb_moe.batched_load_balancing_loss(args)
            rz_loss = mb_router.batched_router_zloss(args).sum() if hasattr(mb_router, 'batched_router_zloss') else ce_loss.new_zeros(())
        else:
            lb_loss = ce_loss.new_zeros(())
            rz_loss = ce_loss.new_zeros(())
        total = ce_loss + lb_loss + rz_loss
        return logits, total, ce_loss, lb_loss, rz_loss


def build_model_from_config(cfg: AttentionMoeConfig, vocab_size: int, sequence_len: int):
    d_model = ((cfg.depth * cfg.aspect_ratio + cfg.head_dim - 1) // cfg.head_dim) * cfg.head_dim
    n_heads = d_model // cfg.head_dim
    config = GPTConfig(sequence_len=sequence_len, vocab_size=vocab_size, n_layer=cfg.depth, n_head=n_heads, n_kv_head=n_heads, n_embd=d_model, window_pattern=cfg.window_pattern)
    return AttentionMoeModel(config, cfg)
