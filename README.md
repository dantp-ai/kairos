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

## Results

*Mean ± 95% bootstrap CI over **30 seeds** per configuration. Regenerate the
per-series figures and significance tables with `bash make_figures.sh`, and the
cross-environment summary with `uv run python summary_figure.py`.*

**More computational power → faster, more stable learning.** On CartPole-v1,
giving `c₂` more gradient work per real-time budget monotonically speeds up
learning *and* shrinks seed variance, approaching the unconstrained ceiling:

![Series A — CartPole](assets/seriesA_cartpole.png)

| `c₂` (E) | median steps to solve (≥475) | seed std of final return |
|---|---|---|
| 1 (E=5)  | 127.5k *(53% of seeds solve)* | 73.0 |
| 2 (E=10) | 85k  | 5.8 |
| 4 (E=20) | 50k  | 0.8 |
| 8 (E=40) | 40k  | 0.0 |
| unconstrained | 30k | 0.0 |

The effect holds across tasks but **diminishes past `c₂≈4`** — and on the harder
Acrobot-v1 the largest budget brings no further speed-up and even starts to
destabilize (`c₂=8` leaves ~1 seed unsolved):

![Compute vs steps-to-solve](assets/summary_compute.png)

**How should a fixed budget be spent — more epochs, or more data?** On CartPole
it doesn't matter (everything saturates). On Acrobot it does, and the lesson is
*don't over-cycle a small batch*: the most-epochs / least-data split (`M=2`) is
never best and is clearly worst at the larger budget, while more data per update
(`M=10`) wins there. AUC = mean return over training (higher is better):

| budget | `M=10` (more data, fewer epochs) | `M=5` | `M=2` (less data, more epochs) |
|---|---|---|---|
| `c₂=2` (M·E=50)  | −144 | **−130** | −144 |
| `c₂=4` (M·E=100) | **−125** | −125 | −134 |

*(Final returns converge across the constrained configs — all eventually solve —
so the signal is in learning speed / AUC, not final return.)*

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
