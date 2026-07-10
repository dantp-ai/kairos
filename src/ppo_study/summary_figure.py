"""Cross-environment summary: how computational power (c2) affects how fast PPO
reaches the 'solved' threshold, for Series A on both environments.

Steps-to-threshold is in the same units (environment steps) for both tasks, so
the two can share a y-axis — unlike raw return. Produces one clean panel that
complements the per-series learning-curve plots from analyze.py.

    uv run python -m ppo_study.summary_figure
"""

import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": False})

# env_dir, threshold, label, colour
ENVS = [
    ("experiments/seriesA/results/cartpole", 475, "CartPole-v1 (solved ≥ 475)", "tab:blue"),
    ("experiments/seriesA/results/acrobot", -100, "Acrobot-v1 (solved ≥ −100)", "tab:red"),
]
C2S = [1, 2, 4, 8]


def steps_to_threshold(run_dir, threshold):
    """Per-seed first checkpoint reaching threshold; returns (median, frac_solved)."""
    xs, hits = None, []
    for f in sorted(os.listdir(run_dir)):
        if not f.endswith(".txt") or f.startswith("stats"):
            continue
        d = np.loadtxt(os.path.join(run_dir, f))
        xs = d[0]
        hits.append(xs[np.argmax(d[1] >= threshold)] if np.any(d[1] >= threshold) else np.nan)
    hits = np.array(hits)
    solved = ~np.isnan(hits)
    median = np.median(hits[solved]) if solved.any() else np.nan
    return median, solved.mean()


def main():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for env_dir, thr, label, colour in ENVS:
        ys, fracs = [], []
        for c2 in C2S:
            m, frac = steps_to_threshold(os.path.join(env_dir, f"c2_{c2}"), thr)
            ys.append(m)
            fracs.append(frac)
        ax.plot(C2S, ys, "o-", color=colour, lw=2, label=label)
        # annotate configs where not every seed solved
        for c2, y, frac in zip(C2S, ys, fracs):
            if frac < 1.0 and not np.isnan(y):
                ax.annotate(f"{frac*100:.0f}% solved", (c2, y), textcoords="offset points",
                            xytext=(6, 8), fontsize=8, color=colour)
        # unconstrained ceiling for this env
        base, _ = steps_to_threshold(os.path.join(env_dir, "baseline"), thr)
        ax.axhline(base, color=colour, ls="--", lw=1, alpha=0.7)

    ax.set_xscale("log", base=2)
    ax.set_xticks(C2S)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Computational power  c₂   (gradient work per real-time budget, M·E = 25·c₂)")
    ax.set_ylabel("Median env-steps to reach 'solved'")
    ax.set_title("More compute → faster solving, with diminishing returns (30 seeds)")
    ax.legend(title="dashed = unconstrained ceiling", fontsize=9)
    fig.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    out = "results/figures/summary_compute.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
