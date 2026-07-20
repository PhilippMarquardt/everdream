"""Verifier functions: shared scoring primitives for RL rewards and evaluation.

Each factory takes a params dict and returns a callable with the TRL reward
signature: fn(prompts, completions, **kwargs) -> list[float | None].
Extra dataset columns (e.g. "ground_truth", "schema") arrive via kwargs as
per-sample lists. Returning None for a sample excludes that verifier for it.

The same functions serve as GRPO reward functions (posttraining) and as
generation metrics (evaluation suites), so train-time rewards and eval-time
metrics can never drift apart.

Binary rewards give zero gradient when a whole group succeeds or fails
uniformly (all advantages equal), so most verifiers here grade partial credit.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerifierSpec:
    name: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


# Backwards-compatible alias for posttraining configs.
RewardSpec = VerifierSpec

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def _completion_text(completion) -> str:
    # Chat datasets yield [{"role": ..., "content": ...}]; plain datasets yield str.
    if isinstance(completion, list):
        completion = "".join(m.get("content", "") for m in completion)
    # Reasoning models (e.g. Qwen3) emit <think>...</think> before the answer;
    # only the post-thinking text is scored.
    return _THINK_RE.sub("", completion).strip()


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


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_answer(text: str) -> str | None:
    """Final answer from a math completion: \\boxed{}, '#### x', else last number."""
    for pattern in (_BOXED_RE, _HASH_ANSWER_RE):
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1] if numbers else None


def _parse_number(s: str) -> float | None:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
        if m and float(m.group(2)) != 0:
            return float(m.group(1)) / float(m.group(2))
        return None


def math_answer(params: dict):
    """1.0 if the final answer matches the per-sample "ground_truth" numerically.

    Per-sample "lower_limit"/"upper_limit" columns (MedCalc-style) switch the
    check to range containment; otherwise relative tolerance against the truth.
    Partial credit params["format_credit"] (default 0.1) when an answer is
    extractable but wrong — keeps group advantages alive early in training.
    """
    format_credit = params.get("format_credit", 0.1)
    rel_tol = params.get("rel_tol", 1e-4)

    def reward(prompts, completions, **kwargs):
        truths = kwargs.get("ground_truth")
        if truths is None:
            return [None] * len(completions)
        n = len(completions)
        lowers = kwargs.get("lower_limit") or [None] * n
        uppers = kwargs.get("upper_limit") or [None] * n
        scores = []
        for c, truth, lo, hi in zip(completions, truths, lowers, uppers):
            answer = extract_final_answer(_completion_text(c))
            if answer is None:
                scores.append(0.0)
                continue
            a, t = _parse_number(answer), _parse_number(str(truth))
            if a is not None and lo is not None and hi is not None:
                lo_f, hi_f = _parse_number(str(lo)), _parse_number(str(hi))
                if lo_f is not None and hi_f is not None:
                    scores.append(1.0 if lo_f <= a <= hi_f else format_credit)
                    continue
            if a is not None and t is not None:
                close = abs(a - t) <= rel_tol * max(1.0, abs(t))
                scores.append(1.0 if close else format_credit)
            else:
                # String fallback (classification labels etc.): normalize case,
                # whitespace, and space/underscore variants.
                norm = lambda s: s.strip().casefold().replace(" ", "_")
                scores.append(1.0 if norm(answer) == norm(str(truth)) else format_credit)
        return scores

    return reward


def exact_match(params: dict):
    """1.0 if the completion equals the per-sample "ground_truth" string (stripped)."""
    casefold = params.get("casefold", False)

    def norm(s: str) -> str:
        s = s.strip()
        return s.casefold() if casefold else s

    def reward(prompts, completions, **kwargs):
        truths = kwargs.get("ground_truth")
        if truths is None:
            return [None] * len(completions)
        return [1.0 if norm(_completion_text(c)) == norm(str(t)) else 0.0 for c, t in zip(completions, truths)]

    return reward


def contains(params: dict):
    """1.0 if the per-sample "ground_truth" string appears in the completion."""
    casefold = params.get("casefold", True)

    def reward(prompts, completions, **kwargs):
        truths = kwargs.get("ground_truth")
        if truths is None:
            return [None] * len(completions)
        scores = []
        for c, t in zip(completions, truths):
            text, needle = _completion_text(c), str(t)
            if casefold:
                text, needle = text.casefold(), needle.casefold()
            scores.append(1.0 if needle in text else 0.0)
        return scores

    return reward


VERIFIER_REGISTRY = {
    "math_answer": math_answer,
    "json_valid": json_valid,
    "json_schema": json_schema,
    "json_match": json_match,
    "format_regex": format_regex,
    "length_penalty": length_penalty,
    "exact_match": exact_match,
    "contains": contains,
}

# Backwards-compatible alias.
REWARD_REGISTRY = VERIFIER_REGISTRY


def build_verifiers(specs: list[VerifierSpec]):
    funcs, weights = [], []
    for spec in specs:
        if spec.name not in VERIFIER_REGISTRY:
            raise ValueError(f"Unknown verifier '{spec.name}'. Available: {sorted(VERIFIER_REGISTRY)}")
        fn = VERIFIER_REGISTRY[spec.name](spec.params)
        fn.__name__ = spec.name  # TRL logs per-reward metrics under this name
        funcs.append(fn)
        weights.append(spec.weight)
    return funcs, weights


# Backwards-compatible alias.
build_reward_funcs = build_verifiers
