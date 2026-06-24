"""Analyse and plot a series of PPO runs.

A *series* is a directory whose immediate sub-directories are individual
configurations (one curve each), e.g. ``results/seriesA_cartpole/{baseline,
c2_1,c2_2,...}``. Each configuration directory holds one ``{seed}.txt`` per run
(two rows: checkpoint steps, average episodic return) and an optional
``stats.txt`` (the JSON config) used for nicer legend labels.

Produces, for the whole series:
  * an overlaid learning-curve plot with mean +/- 95% bootstrap CI bands and the
    unconstrained baseline drawn as a dashed black ceiling line;
  * a summary table (final return, AUC, steps-to-threshold, cross-seed std);
  * a matrix of pairwise Welch t-test p-values on the per-seed final return.

Example:
    uv run python analyze.py results/seriesA_cartpole \
        --title "Series A - CartPole" --threshold 475 \
        --save results/figures/seriesA_cartpole.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Explicitly disable LaTeX rendering so the script runs without a TeX install.
plt.rcParams.update({"text.usetex": False})


def load_run_dir(run_dir):
    """Load every ``{seed}.txt`` in ``run_dir``.

    Returns ``(xs, ys)`` where ``xs`` is the shared checkpoint-step axis and
    ``ys`` has shape ``(n_seeds, n_checkpoints)``. Returns ``None`` if empty.
    """
    xs = None
    rows = []
    for fname in sorted(os.listdir(run_dir)):
        if not fname.endswith(".txt") or fname.startswith("stats"):
            continue
        data = np.loadtxt(os.path.join(run_dir, fname))
        xs = data[0]
        rows.append(data[1])
    if not rows:
        return None
    return xs, np.vstack(rows)


def label_for(run_dir, name):
    """Build a legend label, enriched from stats.txt when available."""
    stats_path = os.path.join(run_dir, "stats.txt")
    if os.path.isfile(stats_path):
        try:
            cfg = json.load(open(stats_path))
            if not cfg.get("constrained", False):
                return f"{name} (unconstrained)"
            return (
                f"c2={cfg.get('c2')}, M={cfg.get('M')}, E={cfg.get('E')} "
                f"(M*E={cfg.get('M') * cfg.get('E')})"
            )
        except Exception:
            pass
    return name


def bootstrap_ci(ys, n_resamples=2000, ci=0.95, seed=0):
    """Per-checkpoint mean and bootstrap CI across seeds.

    ``ys``: (n_seeds, n_checkpoints). Returns (mean, lo, hi), each (n_checkpoints,).
    """
    mean = ys.mean(axis=0)
    if ys.shape[0] < 3:  # too few seeds to bootstrap meaningfully
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    res = stats.bootstrap(
        (ys,),
        np.mean,
        axis=0,
        n_resamples=n_resamples,
        confidence_level=ci,
        random_state=rng,
    )
    return mean, res.confidence_interval.low, res.confidence_interval.high


def final_return_per_seed(ys, frac=0.1):
    """Mean return over the last ``frac`` of checkpoints, per seed -> (n_seeds,)."""
    k = max(1, int(round(ys.shape[1] * frac)))
    return ys[:, -k:].mean(axis=1)


def auc_per_seed(xs, ys):
    """Mean return over all checkpoints (normalised AUC), per seed -> (n_seeds,).

    For evenly-spaced checkpoints this equals the trapezoidal area under the
    learning curve divided by its width, so it summarises whole-training
    performance and discriminates configs even when final returns converge.
    """
    return ys.mean(axis=1)


def steps_to_threshold(xs, ys, threshold):
    """First checkpoint step reaching ``threshold`` per seed; NaN if never."""
    out = []
    for y in ys:
        idx = np.argmax(y >= threshold) if np.any(y >= threshold) else -1
        out.append(xs[idx] if idx >= 0 else np.nan)
    return np.array(out)


def fmt_ci(vals):
    m = np.nanmean(vals)
    if len(vals) >= 3:
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        return f"{m:8.1f} [{lo:7.1f}, {hi:7.1f}]"
    return f"{m:8.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_dir", help="Directory of config sub-dirs.")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Return threshold for steps-to-threshold (e.g. 475 / -100).",
    )
    parser.add_argument("--xlabel", default="Environment steps")
    parser.add_argument(
        "--ylabel", default="Average episodic return (per checkpoint)"
    )
    parser.add_argument("--save", default=None, help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    subdirs = sorted(
        d
        for d in os.listdir(args.series_dir)
        if os.path.isdir(os.path.join(args.series_dir, d))
    )
    # Draw the baseline last so its dashed line sits on top.
    subdirs = [d for d in subdirs if d != "baseline"] + (
        ["baseline"] if "baseline" in subdirs else []
    )

    runs = {}
    for name in subdirs:
        loaded = load_run_dir(os.path.join(args.series_dir, name))
        if loaded is not None:
            runs[name] = loaded
    if not runs:
        raise SystemExit(f"No run data found under {args.series_dir}")

    # ---- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    constrained = [n for n in runs if n != "baseline"]
    colors = {n: cmap(i / max(1, len(constrained) - 1)) for i, n in enumerate(constrained)}

    for name, (xs, ys) in runs.items():
        label = label_for(os.path.join(args.series_dir, name), name)
        mean, lo, hi = bootstrap_ci(ys)
        if name == "baseline":
            ax.plot(xs, mean, color="black", ls="--", lw=2,
                    label=f"{label} [ceiling]", zorder=10)
        else:
            ax.plot(xs, mean, color=colors[name], lw=2, label=label)
            ax.fill_between(xs, lo, hi, color=colors[name], alpha=0.18)

    if args.threshold is not None:
        ax.axhline(args.threshold, color="grey", ls=":", lw=1)
    if args.title:
        ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=args.dpi)
        print(f"Saved figure -> {args.save}")

    # ---- Summary table --------------------------------------------------
    n_seeds = {n: ys.shape[0] for n, (xs, ys) in runs.items()}
    print(f"\n=== Summary: {args.series_dir} (seeds: {n_seeds}) ===")
    header = f"{'config':14s} {'final return [95% CI]':30s} {'AUC':10s} {'std':7s}"
    if args.threshold is not None:
        header += f" {'steps@thr (median, %solved)':28s}"
    print(header)
    finals = {}
    for name, (xs, ys) in runs.items():
        fr = final_return_per_seed(ys)
        finals[name] = fr
        auc = auc_per_seed(xs, ys)
        line = f"{name:14s} {fmt_ci(fr):30s} {np.mean(auc):8.1f}  {np.std(fr):6.1f}"
        if args.threshold is not None:
            s = steps_to_threshold(xs, ys, args.threshold)
            pct = 100 * np.mean(~np.isnan(s))
            med = np.nanmedian(s) if np.any(~np.isnan(s)) else float("nan")
            line += f"  {med:>12.0f}   {pct:4.0f}%"
        print(line)

    # ---- Pairwise Welch t-tests on per-seed final return ----------------
    names = list(runs.keys())
    if len(names) > 1 and min(n_seeds.values()) >= 2:
        print("\nPairwise Welch t-test p-values (per-seed final return):")
        print(" " * 14 + "".join(f"{n[:12]:>13s}" for n in names))
        for a in names:
            row = f"{a:14s}"
            for b in names:
                if a == b:
                    row += f"{'-':>13s}"
                else:
                    p = stats.ttest_ind(finals[a], finals[b], equal_var=False).pvalue
                    row += f"{p:13.3f}"
            print(row)

    plt.close(fig)


if __name__ == "__main__":
    main()
