# Series E - minibatch-size sweep at fixed compute (`batch_size`)

## Impact variable

Minibatch size at a fixed real-time budget: sweep `batch_size`.

## Rationale

Study SGD granularity at fixed compute.
Hold `batch_size·M·E = 4000` so compute (`busy_steps = 800`) and samples processed per burst (`E·M·batch_size = 4000`) are identical everywhere.
Only the minibatch size, and thus the per-burst gradient-step count `E·M`, changes.
Fix `c₂ = 2`.

## Design matrix

Fixed: `C = T = 2000`, `c = 0.1`, `tau = 40`, `c₂ = 2`, `epsilon = 0.2`.
Plus an unconstrained `baseline` ceiling.

| `batch_size` | N | M | E | grad steps/burst `E·M` | `M·bs` |
|---|---|---|---|---|---|
| 50 | 40 | 10 | 8 | 80 | 500 |
| 100 | 20 | 5 | 8 | 40 | 500 |
| 200 | 10 | 5 | 4 | 20 | 1000 |
| 400 | 5 | 2 | 5 | 10 | 800 |

Throughout: `busy_steps = ⌊0.4·bs·M·E / c₂⌋ = ⌊0.2·4000⌋ = 800` and `usable = T - busy_steps = 1200`.
All rows satisfy `busy_steps ≤ T`, `M·batch_size ≤ usable = 1200` (500, 500, 1000, 800), `M ≤ N`.

## Expected outcomes

The classic SGD bias/variance-of-gradient tradeoff.
Small batches (50) take more, noisier steps (faster early, less stable); large batches (400) take fewer, smoother steps (stable, slower).
Predict an intermediate optimum around 100-200, or that PPO is fairly robust across this range on CartPole.
This is the most conventional of the sweeps; it is included as a control that the fixed-compute isolation behaves as ML theory predicts.

## Run

```bash
# CartPole (30 seeds, 10 parallel)
uv run python -m ppo_study.run_experiment experiments/seriesE/config/cartpole.json \
    --n-runs 30 --n-proc 10 --root-dir experiments/seriesE/results/cartpole
# Acrobot
uv run python -m ppo_study.run_experiment experiments/seriesE/config/acrobot.json \
    --n-runs 30 --n-proc 10 --root-dir experiments/seriesE/results/acrobot

# Figures + summary tables
uv run python -m ppo_study.analyze experiments/seriesE/results/cartpole \
    --title "Series E - CartPole-v1" --threshold 475 \
    --save experiments/seriesE/figures/cartpole.png
uv run python -m ppo_study.analyze experiments/seriesE/results/acrobot \
    --title "Series E - Acrobot-v1" --threshold -100 --robust \
    --save experiments/seriesE/figures/acrobot.png

# Copy the published figures into the tracked assets/ dir (embedded in Results above)
cp experiments/seriesE/figures/cartpole.png assets/seriesE_cartpole.png
cp experiments/seriesE/figures/acrobot.png  assets/seriesE_acrobot.png
```

Outputs (`results/`, `figures/`) are gitignored; only `config/` and this README are tracked.
