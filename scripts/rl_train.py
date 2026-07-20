"""GRPO post-training entry point.

Single GPU:
    python scripts/rl_train.py --config configs/rl_grpo_json.yaml

Multi GPU (single machine):
    accelerate launch --num_processes 4 scripts/rl_train.py --config configs/rl_grpo_json.yaml
"""
import argparse

from everdream.posttraining.config import load_rl_config
from everdream.posttraining.grpo import train


def main():
    parser = argparse.ArgumentParser(description="GRPO post-training")
    parser.add_argument("--config", required=True, help="Path to RL YAML config")
    args = parser.parse_args()
    train(load_rl_config(args.config))


if __name__ == "__main__":
    main()
