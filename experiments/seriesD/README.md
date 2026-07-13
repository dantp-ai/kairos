# Series D - update-cadence sweep (`T = C`)

## Impact variable

Update cadence / buffer size: sweep `T = C` (how often the agent updates and how much it collects between bursts).

## Rationale

Sweep how often the agent updates and how much it collects between bursts.
Because `C/N = batch_size`, `busy_steps` is independent of `T`, so this sweep isolates update frequency and the data-drop fraction (`1000/T`) with the compute cost left untouched.
Fix `c₂ = 2, M = 5, E = 10`, and set both `C` and `T` to the same value so every row spends the same `busy_steps = 1000`.

## Design matrix

Fixed: `batch_size = 100`, `c = 0.1`, `tau = 40`, `M = 5`, `E = 10`, `c₂ = 2`.
Plus an unconstrained `baseline` ceiling.

| `T = C` | N | usable | drop fraction (`1000/T`) | bursts over 150k |
|---|---|---|---|---|
| 1500 | 15 | 500 | 67% | 100 |
| 2000 | 20 | 1000 | 50% | 75 |
| 3000 | 30 | 2000 | 33% | 50 |
| 4000 | 40 | 3000 | 25% | 37 |
| 6000 | 60 | 5000 | 17% | 25 |

All rows satisfy `busy_steps = 1000 ≤ T`, `M·batch_size = 500 ≤ usable`, `M = 5 ≤ N`.

## Expected outcomes

Expect an interior sweet spot.
Small `T` updates often but drops 67% of the data onto a tiny buffer, so updates are noisy and unstable.
Large `T` drops little but updates rarely and on staler data, so it is slow per env step.
CartPole likely mostly solves across the sweep, with the differences showing up in speed and variance; Acrobot is more sensitive and should favour a moderate `T`.
Caveat: larger `T` means fewer total bursts over the fixed `num_steps`, so slower learning there could be cadence OR simply reduced total gradient work.
A work-matched follow-up (scale `num_steps` proportional to `T`) disambiguates the two.

## Run

```bash
# CartPole (30 seeds, 10 parallel)
uv run python -m ppo_study.run_experiment experiments/seriesD/config/cartpole.json \
    --n-runs 30 --n-proc 10 --root-dir experiments/seriesD/results/cartpole
# Acrobot
uv run python -m ppo_study.run_experiment experiments/seriesD/config/acrobot.json \
    --n-runs 30 --n-proc 10 --root-dir experiments/seriesD/results/acrobot

# Figures + summary tables
uv run python -m ppo_study.analyze experiments/seriesD/results/cartpole \
    --title "Series D - CartPole-v1" --threshold 475 \
    --save experiments/seriesD/figures/cartpole.png
uv run python -m ppo_study.analyze experiments/seriesD/results/acrobot \
    --title "Series D - Acrobot-v1" --threshold -100 --robust \
    --save experiments/seriesD/figures/acrobot.png

# Copy the published figures into the tracked assets/ dir (embedded in Results above)
cp experiments/seriesD/figures/cartpole.png assets/seriesD_cartpole.png
cp experiments/seriesD/figures/acrobot.png  assets/seriesD_acrobot.png
```

Outputs (`results/`, `figures/`) are gitignored; only `config/` and this README are tracked.
