# Series F - environment-speed sweep (`tau` with a decoupled real-time cost model)

## Impact variable

Environment speed `tau`: sweep the env-step time while holding gradient work fixed.

## Rationale

Make "the environment doesn't wait" a real knob.
In the original cost model `tau` cancels out of `busy_steps = ⌊H·c/c₂⌋`, so env speed has no effect on how much data is dropped.
Setting a reference clock `tau_ref` fixes the burst's compute time independently of the env's own step time, so a faster environment (smaller `tau`) leaves a bigger busy window and drops more of the samples gathered during a burst.
The point `tau = tau_ref = 40` reproduces the original model exactly.

## Cost-model change

`run()` gains an optional `tau_ref` (default `None`), a backward-compatible addition:

- `tau_ref is None` -> `L = H·(c·tau)/c₂`, identical to before (Series A/B/C are unaffected, byte-for-byte).
- `tau_ref` set -> `L = H·(c·tau_ref)/c₂`, which decouples the burst's compute time from the env clock.

In both cases `busy_steps = ⌊L/tau⌋`.
With `tau_ref` set this becomes `busy_steps = ⌊H·c·tau_ref/(c₂·tau)⌋`, so a smaller `tau` yields a larger busy window.

## Design matrix

Fixed across constrained rows: `C = T = 2000`, `batch_size = 100`, `c = 0.1`, `c₂ = 2`, `M = 5`, `E = 10`, `tau_ref = 40`.
With these, `busy_steps = ⌊40000/tau⌋`.
Plus an unconstrained `baseline` ceiling (no `tau_ref`).

| `tau` | busy_steps `= ⌊40000/tau⌋` | usable | drop fraction |
|---|---|---|---|
| 160 | 250 | 1750 | 12.5% |
| 80 | 500 | 1500 | 25% |
| 40 | 1000 | 1000 | 50% |
| 30 | 1333 | 667 | 67% |
| baseline | - | - | - |

All constrained rows satisfy `busy_steps ≤ T`, `M·batch_size = 500 ≤ usable`, `M = 5 ≤ N = 20`.

## Expected outcomes

Monotone degradation as `tau` falls: a slower env (large `tau`) keeps a small busy window and drops little data, while a faster env (small `tau`) grows the busy window and drops more, giving slower and less stable learning.
Key qualitative finding: below `tau ≈ 27` the config is infeasible (`usable < M·batch_size = 500`), i.e. a fast enough environment permanently outruns a fixed-compute learner and Constraint 2 / the usable-samples assert fires.
Because `tau` and `c₂` act through the same `busy_steps`, results should mirror Series C: on CartPole the effect is small (even 50% dropping is tolerable), while on data-hungry Acrobot dropped experience matters more.

## Run

```bash
# CartPole (30 seeds, 10 parallel)
uv run python -m ppo_study.run_experiment experiments/seriesF/config/cartpole.json \
    --n-runs 30 --n-proc 10 --root-dir experiments/seriesF/results/cartpole
# Acrobot
uv run python -m ppo_study.run_experiment experiments/seriesF/config/acrobot.json \
    --n-runs 30 --n-proc 10 --root-dir experiments/seriesF/results/acrobot

# Figures + summary tables
uv run python -m ppo_study.analyze experiments/seriesF/results/cartpole \
    --title "Series F - CartPole-v1" --threshold 475 \
    --save experiments/seriesF/figures/cartpole.png
uv run python -m ppo_study.analyze experiments/seriesF/results/acrobot \
    --title "Series F - Acrobot-v1" --threshold -100 --robust \
    --save experiments/seriesF/figures/acrobot.png

# Copy the published figures into the tracked assets/ dir (embedded in Results above)
cp experiments/seriesF/figures/cartpole.png assets/seriesF_cartpole.png
cp experiments/seriesF/figures/acrobot.png  assets/seriesF_acrobot.png
```

Outputs (`results/`, `figures/`) are gitignored; only `config/` and this README are tracked.
