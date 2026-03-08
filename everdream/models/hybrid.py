from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from everdream.common import COMPUTE_DTYPE, get_dist_info, print0
from everdream.config.schema import HybridModelConfig
from everdream.models.dense import Linear, apply_rotary_emb
from everdream.optim import DistMuonAdamW, MuonAdamW

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
except ImportError:
    fla_chunk_gated_delta_rule = None


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class RMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        input_dtype = x.dtype
        x = F.rms_norm(x.float(), (x.size(-1),), eps=self.eps)
        x = (self.weight * x).to(input_dtype)
        return (x * F.silu(gate.float())).to(input_dtype)


class DenseMLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_up = Linear(d_model, 8 * d_model, bias=False)
        self.down = Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class GatedDeltaNetMixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, chunk_size: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        kv_dim = n_heads * head_dim
        self.in_proj = Linear(d_model, kv_dim * 4, bias=False)
        self.ba_proj = Linear(d_model, n_heads * 2, bias=False)
        conv_dim = kv_dim * 3
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=4, bias=False, groups=conv_dim, padding=3)
        self.dt_bias = nn.Parameter(torch.ones(n_heads))
        A = torch.empty(n_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNormGated(head_dim)
        self.o_proj = Linear(kv_dim, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        H, d = self.n_heads, self.head_dim
        qkvz = self.in_proj(x)
        q, k, v, z = qkvz.split([H * d, H * d, H * d, H * d], dim=-1)
        ba = self.ba_proj(x)
        b, a = ba.split([H, H], dim=-1)
        qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
        qkv = F.silu(self.conv1d(qkv)[:, :, :T]).transpose(1, 2)
        q, k, v = qkv.split([H * d, H * d, H * d], dim=-1)
        q = q.reshape(B, T, H, d)
        k = k.reshape(B, T, H, d)
        v = v.reshape(B, T, H, d)
        z = z.reshape(B, T, H, d)
        beta = b.sigmoid()
        A = self.A_log.float().clamp(min=-2.3).exp()
        g = -A * F.softplus(a.float() + self.dt_bias)
        if fla_chunk_gated_delta_rule is None:
            raise RuntimeError('flash-linear-attention is required for the hybrid DeltaNet model')
        y, _ = fla_chunk_gated_delta_rule(q.float(), k.float(), v.float(), g=g, beta=beta, chunk_size=self.chunk_size, use_qk_l2norm_in_kernel=True)
        y = self.norm(y.reshape(-1, d), z.reshape(-1, d)).reshape(B, T, -1)
        return self.o_proj(y)


class GatedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.c_q = Linear(d_model, n_heads * head_dim, bias=False)
        self.c_k = Linear(d_model, n_heads * head_dim, bias=False)
        self.c_v = Linear(d_model, n_heads * head_dim, bias=False)
        self.c_z = Linear(d_model, n_heads * head_dim, bias=False)
        self.c_beta = Linear(d_model, n_heads, bias=False)
        self.c_proj = Linear(n_heads * head_dim, d_model, bias=False)
        self.norm = RMSNormGated(head_dim)

    def forward(self, x, cos_sin):
        B, T, _ = x.shape
        H, d = self.n_heads, self.head_dim
        q = self.c_q(x).view(B, T, H, d)
        k = self.c_k(x).view(B, T, H, d)
        v = self.c_v(x).view(B, T, H, d)
        z = self.c_z(x).view(B, T, H, d)
        beta = torch.sigmoid(self.c_beta(x)).unsqueeze(-1)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        y = self.norm((beta * y).reshape(-1, d), z.reshape(-1, d)).reshape(B, T, -1)
        return self.c_proj(y)


class DeltaNetBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int, chunk_size: int):
        super().__init__()
        self.mixer = GatedDeltaNetMixer(d_model, n_heads, head_dim, chunk_size)
        self.mlp = DenseMLP(d_model)

    def forward(self, x):
        x = x + self.mixer(norm(x))
        x = x + self.mlp(norm(x))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.attn = GatedSelfAttention(d_model, n_heads, head_dim)
        self.mlp = DenseMLP(d_model)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class HybridDeltaNetModel(nn.Module):
    def __init__(self, cfg: HybridModelConfig, vocab_size: int, sequence_len: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            AttentionBlock(cfg.d_model, cfg.n_heads, cfg.d_model // cfg.n_heads) if cfg.layer_pattern[i] == 'A' else DeltaNetBlock(cfg.d_model, cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.chunk_size)
            for i in range(cfg.depth)
        ])
        self.lm_head = Linear(cfg.d_model, vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(cfg.depth))
        self.x0_lambdas = nn.Parameter(torch.zeros(cfg.depth))
        cos, sin = self._precompute_rotary_embeddings(sequence_len * 10, cfg.d_model // cfg.n_heads)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().to(COMPUTE_DTYPE), freqs.sin().to(COMPUTE_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.cfg.d_model**-0.5
        for block in self.blocks:
            if isinstance(block, AttentionBlock):
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_z.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_beta.weight)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            else:
                torch.nn.init.uniform_(block.mixer.in_proj.weight, -s, s)
                torch.nn.init.zeros_(block.mixer.o_proj.weight)
                torch.nn.init.zeros_(block.mixer.ba_proj.weight)
            torch.nn.init.uniform_(block.mlp.gate_up.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.down.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

    def estimate_flops(self):
        total_active = 0
        d = self.cfg.d_model
        h = self.cfg.n_heads
        q = d // h
        for c in self.cfg.layer_pattern:
            total_active += 17 * d * d + (d * h if c == 'A' else 2 * d * h)
        total_active += d * self.vocab_size
        attn_flops = sum(12 * h * q * self.sequence_len for i in range(self.cfg.depth) if self.cfg.layer_pattern[i] == 'A')
        delta_flops = sum(6 * h * q * self.sequence_len for i in range(self.cfg.depth) if self.cfg.layer_pattern[i] == 'D')
        return 6 * total_active + attn_flops + delta_flops

    def num_scaling_params(self):
        total = sum(p.numel() for p in self.parameters())
        return {'total': total, 'active': total}

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.cfg.d_model
        ddp, *_ = get_dist_info()
        matrix_params = list(self.blocks.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ?1/v({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
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
        x = self.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin) if isinstance(block, AttentionBlock) else block(x)
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 20 * torch.tanh(logits / 20)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


def build_model_from_config(cfg: HybridModelConfig, vocab_size: int, sequence_len: int):
    return HybridDeltaNetModel(cfg, vocab_size=vocab_size, sequence_len=sequence_len)
