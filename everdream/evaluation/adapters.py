"""Model adapters: one generation/scoring interface over both model worlds.

EverdreamAdapter wraps the native GPT checkpoints (pre/mid-training);
HFAdapter wraps any HuggingFace causal LM (posttraining). Eval tasks only
talk to the adapter, so every suite runs unchanged across phases.
"""
from __future__ import annotations

import torch

from .config import GenParams


class ModelAdapter:
    def generate(self, prompts: list, gen: GenParams) -> list[str]:
        """prompts: list of strings, or lists of chat messages when the task data is chat-style.
        Returns completions only (no prompt echo)."""
        raise NotImplementedError

    @property
    def lm_model(self):
        """Raw model for loss-based tasks (bpb, core); None if unsupported."""
        return None


class EverdreamAdapter(ModelAdapter):
    def __init__(self, model, tokenizer, device):
        # Unwrap torch.compile so .generate and eval-mode flags hit the real module.
        self.model = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.tokenizer = tokenizer
        self.device = device

    @property
    def lm_model(self):
        return self.model

    def _to_text(self, prompt) -> str:
        if isinstance(prompt, list):  # chat messages: no chat template in base models, join contents
            return "\n\n".join(m["content"] for m in prompt)
        return prompt

    @torch.no_grad()
    def generate(self, prompts: list, gen: GenParams) -> list[str]:
        from everdream.eval import disable_fp8

        was_training = self.model.training
        self.model.eval()
        eos_id = self.tokenizer.get_eos_token_id()
        completions = []
        try:
            with disable_fp8(self.model):
                for i, prompt in enumerate(prompts):
                    ids = self.tokenizer.encode(self._to_text(prompt), prepend=self.tokenizer.get_bos_token_id())
                    generated = []
                    stream = self.model.generate(
                        ids,
                        max_tokens=gen.max_new_tokens,
                        temperature=gen.temperature,
                        top_k=gen.top_k or None,
                        seed=gen.seed + i,
                    )
                    for token in stream:
                        if token == eos_id:
                            break
                        generated.append(token)
                    completions.append(self.tokenizer.decode(generated))
        finally:
            if was_training:
                self.model.train()
        return completions


class HFAdapter(ModelAdapter):
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(cls, name_or_path, device=None, torch_dtype="bfloat16", attn_implementation="sdpa", trust_remote_code=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, torch_dtype) if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, torch_dtype=dtype, attn_implementation=attn_implementation, trust_remote_code=trust_remote_code
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        return cls(model, tokenizer, device)

    def _render(self, prompt) -> str:
        if isinstance(prompt, list):
            return self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return prompt

    @torch.no_grad()
    def generate(self, prompts: list, gen: GenParams) -> list[str]:
        was_training = self.model.training
        self.model.eval()
        texts = [self._render(p) for p in prompts]
        # Left padding so all sequences end at the generation boundary.
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        completions = []
        try:
            for start in range(0, len(texts), gen.batch_size):
                batch = texts[start : start + gen.batch_size]
                enc = self.tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True).to(self.device)
                out = self.model.generate(
                    **enc,
                    max_new_tokens=gen.max_new_tokens,
                    do_sample=gen.temperature > 0,
                    temperature=gen.temperature if gen.temperature > 0 else None,
                    top_k=gen.top_k if gen.temperature > 0 and gen.top_k > 0 else None,
                    top_p=gen.top_p if gen.temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                new_tokens = out[:, enc["input_ids"].shape[1] :]
                completions.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        finally:
            self.tokenizer.padding_side = prev_side
            if was_training:
                self.model.train()
        return completions
