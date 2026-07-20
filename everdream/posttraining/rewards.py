"""Rewards live in everdream.evaluation.verifiers so RL rewards and eval metrics share one registry."""
from everdream.evaluation.verifiers import (  # noqa: F401
    REWARD_REGISTRY,
    RewardSpec,
    build_reward_funcs,
    extract_json_candidate,
)
