# PPO under real-time computational constraints

We study Proximal Policy Optimization (PPO) when the learner runs on a machine
with finite **computational power** `c₂`. In a real-time system, computing a
learning burst takes wall-clock time `L = H·c·τ/c₂` (where `H = 4·(C/N)·M·E` is
the number of forward/backward passes in a burst). During that time the agent
must keep acting, so two things happen:

1. the newly learned parameters cannot be **deployed** until the burst finishes
   (`tp = t + ⌊L/τ⌋`); and
2. the samples collected **while the learner is busy** (`[t, t + ⌊L/τ⌋]`) cannot
   be ingested by the learner and are dropped.

Modelling **both** effects (earlier versions of this project only modelled #1)
is what makes `c₂` actually affect performance. With the data-drop in place, a
burst learns on only `T − ⌊L/τ⌋` of the `T` samples gathered since the last
burst (see `train_ppo.py`).

## Setup (uv + Python 3.12)

```bash
uv sync          # creates .venv from pyproject.toml
```

Then prefix commands with `uv run`, e.g. `uv run python run_experiment.py ...`.

## A single run

```bash
uv run python train_ppo.py --env-name CartPole-v1 --constrained \
    --C 2000 --T 2000 --batch-size 100 --M 5 --E 10 \
    --c 0.1 --tau 40 --c2 2 --num-steps 150000 --checkpoint 5000 --seed 0
```

Each run writes `<out_dir>/<seed>.txt` (row 0 = checkpoint steps, row 1 = average
episodic return), `<out_dir>/models/<seed>.pth`, and `<out_dir>/stats.txt`.

## Experiment series

Fixed everywhere: `C = T = 2000`, `batch_size = 100` (⇒ `N = 20`), `c = 0.1`,
`τ = 40`, `ε = 0.2`, GAE `γ = 0.99 / λ = 0.95`, 2×64 tanh actor & critic. Each
constrained config keeps the real-time constraint at a **constant 50% data
waste** (`⌊L/τ⌋ = 1000`), so the corrected model applies uniformly.

* **Series A — computational power.** Hold the allocation `M = 5` and let `c₂`
  buy more gradient work: `M·E = 25·c₂`, i.e. `E = 5·c₂`, for `c₂ ∈ {1,2,4,8}`.
  Question: *does a more powerful machine learn better?* (config:
  `config/seriesA_{cartpole,acrobot}.json`)
* **Series B — how to spend the budget.** At `M·E = 25·c₂`, vary the epochs↔
  minibatches split with `M` common across columns `{10, 5, 2}` for
  `c₂ ∈ {2, 4}`. Question: *more epochs over the same data, or more data per
  update?* (config: `config/seriesB_{cartpole,acrobot}.json`)

Both series include an unconstrained `baseline` (the performance ceiling). The
`c₂∈{2,4}, M=5` rows of Series A are identical to Series B's middle column, so
the two series form one consistent grid.

## Reproduce

```bash
# 10 seeds per config across 4 sweeps (≈15-20 min on a 12-core machine)
for s in seriesA_cartpole seriesA_acrobot seriesB_cartpole seriesB_acrobot; do
  uv run python run_experiment.py config/$s.json --n-runs 10 --n-proc 10 --root-dir results
done

# learning-curve figures (mean ± 95% bootstrap CI) + summary & significance tables
bash make_figures.sh
```

Scaling to 30 seeds is seamless: `--n-runs 30` reproduces the first 10 seeds and
skips configs whose `<seed>.txt` already exists.

`analyze.py` can also be pointed at any single series directory:

```bash
uv run python analyze.py results/seriesA_cartpole \
    --title "Series A - CartPole" --threshold 475 --save results/figures/seriesA_cartpole.png
```
