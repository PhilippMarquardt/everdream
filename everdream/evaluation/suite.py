from __future__ import annotations

import time

from everdream.common import print0

from .adapters import ModelAdapter
from .config import EvalSuiteConfig
from .tasks import TASK_REGISTRY, EvalContext


def run_suite(adapter: ModelAdapter, suite: EvalSuiteConfig, ctx: EvalContext | None = None) -> dict:
    ctx = ctx or EvalContext()
    results = {}
    for spec in suite.tasks:
        if spec.type not in TASK_REGISTRY:
            raise ValueError(f"Unknown eval task type '{spec.type}'. Available: {sorted(TASK_REGISTRY)}")
        t0 = time.time()
        results[spec.name] = TASK_REGISTRY[spec.type](adapter, spec, ctx)
        results[spec.name]["seconds"] = round(time.time() - t0, 2)
    return results


def flatten_results(results: dict, prefix: str = "eval/") -> dict:
    """Scalar metrics only, wandb-ready keys: eval/<task>/<metric>."""
    flat = {}
    for task_name, res in results.items():
        for key, value in res.items():
            if isinstance(value, (int, float)) and key != "seconds":
                flat[f"{prefix}{task_name}/{key}"] = value
            elif key == "metrics":
                for m, v in value.items():
                    flat[f"{prefix}{task_name}/{m}"] = v
            elif key == "core_metric":
                flat[f"{prefix}{task_name}/core_metric"] = value
    return flat


def print_results(results: dict, header: str = "eval") -> None:
    print0(f"=== {header} ===")
    for task_name, res in results.items():
        scalars = {k: v for k, v in flatten_results({task_name: res}, prefix="").items()}
        line = " | ".join(f"{k} {v:.4f}" for k, v in scalars.items())
        print0(f"  {task_name}: {line if line else '(no scalar metrics)'} [{res.get('seconds', 0)}s]")
        for sample in res.get("samples", [])[:2]:
            completion = sample["completion"].replace("\n", " ")[:120]
            print0(f"    > {sample['prompt'][:60]!r} -> {completion!r}")
