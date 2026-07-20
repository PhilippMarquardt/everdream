from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .verifiers import VerifierSpec


@dataclass
class GenParams:
    max_new_tokens: int = 256
    temperature: float = 0.0  # 0 = greedy
    top_k: int = 0
    top_p: float = 1.0
    batch_size: int = 8  # batched adapters only; sequential adapters ignore it
    seed: int = 42


@dataclass
class EvalDataSpec:
    source: str = "jsonl"  # jsonl | hf
    path: str = ""
    split: str = "test"
    prompt_field: str = "prompt"
    max_samples: int = -1
    chat: bool = False
    system_prompt: str = ""


@dataclass
class EvalTaskSpec:
    name: str
    type: str  # generation | sample | bpb | core
    data: EvalDataSpec | None = None
    gen: GenParams = field(default_factory=GenParams)
    metrics: list[VerifierSpec] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSuiteConfig:
    name: str = "eval"
    tasks: list[EvalTaskSpec] = field(default_factory=list)


def suite_from_dict(raw: dict) -> EvalSuiteConfig:
    tasks = []
    for t in raw.get("tasks", []):
        t = dict(t)
        data = EvalDataSpec(**t.pop("data")) if "data" in t else None
        gen = GenParams(**t.pop("gen")) if "gen" in t else GenParams()
        metrics = [VerifierSpec(**m) for m in t.pop("metrics", [])]
        tasks.append(EvalTaskSpec(data=data, gen=gen, metrics=metrics, **t))
    if not tasks:
        raise ValueError("Eval suite must define at least one task")
    names = [t.name for t in tasks]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate task names in eval suite: {names}")
    return EvalSuiteConfig(name=raw.get("name", "eval"), tasks=tasks)


def load_eval_suite(path: str | Path) -> EvalSuiteConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return suite_from_dict(raw)
