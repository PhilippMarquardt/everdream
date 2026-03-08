from __future__ import annotations

from dataclasses import asdict, is_dataclass
import os

from everdream.runtime.distributed import DummyWandb


def _to_loggable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_loggable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_loggable(v) for v in obj]
    return obj


def init_wandb(enabled: bool, project: str, run_name: str, config: dict, master_process: bool, mode: str = "online"):
    if not enabled or not master_process:
        return DummyWandb()
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    import wandb
    return wandb.init(project=project, name=run_name, config=_to_loggable(config), mode=mode)
