# everdream
play around

- `everdream/pretraining` — pre/mid-training (nanochat-style distributed trainer, FP8, FA3, weighted parquet mixing)
- `everdream/posttraining` — RL post-training (GRPO via TRL, pluggable verification rewards)
- `everdream/evaluation` — phase-agnostic eval suites (YAML-configured tasks over native or HF models; verifiers shared with RL rewards)

Eval (same suite works for pre/mid/post):

```bash
# native checkpoint                                       # HF / post-RL model
python scripts/eval.py --suite configs/eval_suite_pretrain.yaml \
    --source everdream --config <cfg.yaml> --checkpoint-step <n>
python scripts/eval.py --suite configs/eval_suite_math.yaml --source hf --model out/rl_gsm8k/final
```

In-training: `training.eval_suite: <suite.yaml>` (pretrain config) or `eval.suite: <suite.yaml>` (RL config).

RL quickstart (single machine):

```bash
pip install -e .[rl]
python scripts/prepare_rl_data.py                                                   # GSM8K -> data/gsm8k
python scripts/rl_train.py --config configs/rl_grpo_math.yaml                       # single GPU
accelerate launch --num_processes 4 scripts/rl_train.py --config configs/rl_grpo_math.yaml  # multi GPU
```
