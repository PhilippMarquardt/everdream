"""Verification-style reward functions for GRPO.

Each factory takes a params dict and returns a callable with the TRL reward
signature: fn(prompts, completions, **kwargs) -> list[float | None].
Extra dataset columns (e.g. "ground_truth", "schema") arrive via kwargs as
per-sample lists. Returning None for a sample excludes that reward for it.

Binary rewards give zero gradient when a whole group succeeds or fails
uniformly (all advantages equal), so most rewards here grade partial credit.
"""
from __future__ import annotations

import json
import re

from .config import RewardSpec

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _completion_text(completion) -> str:
    # Chat datasets yield [{"role": ..., "content": ...}]; plain datasets yield str.
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion)
    return completion


def extract_json_candidate(text: str, mode: str) -> str:
    if mode == "fence":
        m = _FENCE_RE.search(text)
        return m.group(1) if m else text
    if mode == "first_object":
        m = _OBJECT_RE.search(text)
        return m.group(0) if m else text
    return text


def _try_parse(text: str, mode: str):
    try:
        return json.loads(extract_json_candidate(text, mode).strip())
    except (json.JSONDecodeError, RecursionError):
        return None


def json_valid(params: dict):
    """1.0 for parseable JSON. Partial credit: 0.25 if it at least looks like JSON."""
    mode = params.get("extract", "none")

    def reward(prompts, completions, **kwargs):
        scores = []
        for c in completions:
            text = _completion_text(c)
            if _try_parse(text, mode) is not None:
                scores.append(1.0)
            else:
                candidate = extract_json_candidate(text, mode).strip()
                scores.append(0.25 if candidate[:1] in "{[" else 0.0)
        return scores

    return reward


def json_schema(params: dict):
    """Graded schema check: 0.4 for valid JSON object, up to 0.6 for required keys.

    Keys come from params["required_keys"], optionally overridden per sample by a
    "schema" dataset column holding a JSON object like {"required": [...]}. If the
    jsonschema package is installed and the per-sample schema is a full JSON Schema,
    it is validated properly (all-or-nothing on top of the parse credit).
    """
    mode = params.get("extract", "none")
    default_keys = list(params.get("required_keys", []))

    try:
        import jsonschema
    except ImportError:
        jsonschema = None

    def reward(prompts, completions, **kwargs):
        schemas = kwargs.get("schema", [None] * len(completions))
        scores = []
        for c, schema in zip(completions, schemas):
            obj = _try_parse(_completion_text(c), mode)
            if not isinstance(obj, dict):
                scores.append(0.0)
                continue
            score = 0.4
            if isinstance(schema, str):
                schema = _try_parse(schema, "none")
            if schema and jsonschema is not None and set(schema) - {"required"}:
                try:
                    jsonschema.validate(obj, schema)
                    score += 0.6
                except jsonschema.ValidationError:
                    pass
            else:
                keys = (schema or {}).get("required", default_keys) if isinstance(schema, dict) else default_keys
                if keys:
                    score += 0.6 * sum(k in obj for k in keys) / len(keys)
                else:
                    score += 0.6
            scores.append(score)
        return scores

    return reward


def json_match(params: dict):
    """Compare parsed completion against a per-sample "ground_truth" column.

    Full match = 1.0; for dicts, partial credit = fraction of ground-truth
    key/value pairs reproduced exactly.
    """
    mode = params.get("extract", "none")

    def reward(prompts, completions, **kwargs):
        truths = kwargs.get("ground_truth")
        if truths is None:
            return [None] * len(completions)
        scores = []
        for c, truth in zip(completions, truths):
            if isinstance(truth, str):
                truth = _try_parse(truth, "none")
            obj = _try_parse(_completion_text(c), mode)
            if obj is None or truth is None:
                scores.append(0.0)
            elif obj == truth:
                scores.append(1.0)
            elif isinstance(obj, dict) and isinstance(truth, dict) and truth:
                scores.append(sum(obj.get(k) == v for k, v in truth.items()) / len(truth))
            else:
                scores.append(0.0)
        return scores

    return reward


def format_regex(params: dict):
    """1.0 if the completion matches params["pattern"] (search by default)."""
    pattern = re.compile(params["pattern"], re.DOTALL)
    full = params.get("fullmatch", False)

    def reward(prompts, completions, **kwargs):
        match = pattern.fullmatch if full else pattern.search
        return [1.0 if match(_completion_text(c)) else 0.0 for c in completions]

    return reward


def length_penalty(params: dict):
    """0 within budget, linearly down to -1 at 2x params["max_chars"]."""
    max_chars = params.get("max_chars", 2048)

    def reward(prompts, completions, **kwargs):
        scores = []
        for c in completions:
            over = len(_completion_text(c)) - max_chars
            scores.append(0.0 if over <= 0 else -min(1.0, over / max_chars))
        return scores

    return reward


REWARD_REGISTRY = {
    "json_valid": json_valid,
    "json_schema": json_schema,
    "json_match": json_match,
    "format_regex": format_regex,
    "length_penalty": length_penalty,
}


def build_reward_funcs(specs: list[RewardSpec]):
    funcs, weights = [], []
    for spec in specs:
        if spec.name not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward '{spec.name}'. Available: {sorted(REWARD_REGISTRY)}")
        fn = REWARD_REGISTRY[spec.name](spec.params)
        fn.__name__ = spec.name  # TRL logs per-reward metrics under this name
        funcs.append(fn)
        weights.append(spec.weight)
    return funcs, weights
