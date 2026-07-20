# everdream
play around

- `everdream/pretraining` — pre/mid-training (nanochat-style distributed trainer, FP8, FA3, weighted parquet mixing)
- `everdream/posttraining` — RL post-training (GRPO via TRL, pluggable verification rewards)

RL quickstart (single machine):

```bash
pip install -e .[rl]
python scripts/rl_train.py --config configs/rl_grpo_json.yaml                      # single GPU
accelerate launch --num_processes 4 scripts/rl_train.py --config configs/rl_grpo_json.yaml  # multi GPU
```
