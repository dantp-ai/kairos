"""Analyse and plot a series of PPO runs.

A *series* is a directory whose immediate sub-directories are individual
configurations (one curve each), e.g. ``results/seriesA_cartpole/{baseline,
c2_1,c2_2,...}``. Each configuration directory holds one ``{seed}.txt`` per run
(two rows: checkpoint steps, average episodic return) and an optional
``stats.txt`` (the JSON config) used for nicer legend labels.

Produces, for the whole series:
  * an overlaid learning-curve plot — mean + 95% bootstrap CI band by default,
    or median + interquartile band with ``--robust`` (cleaner when a few seeds
    diverge) — with the unconstrained baseline as a dashed black ceiling line;
  * a summary table (final return, AUC, steps-to-threshold, cross-seed std);
  * a matrix of pairwise Welch t-test p-values on the per-seed final return.

Example:
    uv run python analyze.py results/seriesA_acrobot \
        --title "Series A - Acrobot" --threshold -100 --robust \
        --save results/figures/seriesA_acrobot.png
"""

import argparse
import json
import os
import warnings

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


def band(ys, robust=False, n_resamples=2000, ci=0.95, seed=0):
    """Central curve + shaded interval across seeds, per checkpoint.

    ``ys``: (n_seeds, n_checkpoints). Returns ``(center, lo, hi)``.

    * default    -> mean + bootstrap percentile CI;
    * ``robust`` -> median + interquartile (25-75) band, which is far cleaner
      than a bootstrap CI when a few seeds diverge (e.g. on Acrobot).
    """
    if robust:
        return (
            np.median(ys, axis=0),
            np.percentile(ys, 25, axis=0),
            np.percentile(ys, 75, axis=0),
        )
    mean = ys.mean(axis=0)
    if ys.shape[0] < 3:  # too few seeds to bootstrap meaningfully
        return mean, mean, mean
    with warnings.catch_warnings():
        # Saturated configs (e.g. every seed at 500) are zero-variance; the
        # percentile method handles them gracefully and we silence the
        # residual numerical notices rather than spam the console.
        warnings.simplefilter("ignore")
        res = stats.bootstrap(
            (ys,),
            np.mean,
            axis=0,
            n_resamples=n_resamples,
            confidence_level=ci,
            method="percentile",
            random_state=np.random.default_rng(seed),
        )
    lo, hi = res.confidence_interval.low, res.confidence_interval.high
    # Degenerate (zero-variance) checkpoints -> collapse the band onto the mean.
    lo = np.where(np.isfinite(lo), lo, mean)
    hi = np.where(np.isfinite(hi), hi, mean)
    return mean, lo, hi


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


def fmt_center(vals, robust=False):
    """Format center [interval]: median [IQR] if robust, else mean [95% CI]."""
    if robust:
        med, lo, hi = np.nanpercentile(vals, [50, 25, 75])
        return f"{med:8.1f} [{lo:7.1f}, {hi:7.1f}]"
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
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Plot median + interquartile band (and report median) instead of "
        "mean + bootstrap CI. Cleaner when a few seeds diverge (e.g. Acrobot).",
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
    band_desc = "median +/- IQR (25-75)" if args.robust else "mean +/- 95% bootstrap CI"
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    constrained = [n for n in runs if n != "baseline"]
    colors = {n: cmap(i / max(1, len(constrained) - 1)) for i, n in enumerate(constrained)}

    for name, (xs, ys) in runs.items():
        label = label_for(os.path.join(args.series_dir, name), name)
        center, lo, hi = band(ys, robust=args.robust)
        if name == "baseline":
            ax.plot(xs, center, color="black", ls="--", lw=2,
                    label=f"{label} [ceiling]", zorder=10)
        else:
            ax.plot(xs, center, color=colors[name], lw=2, label=label)
            ax.fill_between(xs, lo, hi, color=colors[name], alpha=0.18)

    if args.threshold is not None:
        ax.axhline(args.threshold, color="grey", ls=":", lw=1)
    if args.title:
        ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.legend(fontsize=8, loc="best")
    ax.text(0.99, 0.01, f"shaded: {band_desc}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="grey")
    fig.tight_layout()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=args.dpi)
        print(f"Saved figure -> {args.save}")

    # ---- Summary table --------------------------------------------------
    n_seeds = {n: ys.shape[0] for n, (xs, ys) in runs.items()}
    print(f"\n=== Summary: {args.series_dir} (seeds: {n_seeds}) ===")
    center_label = "median [IQR]" if args.robust else "mean [95% CI]"
    auc_center = np.median if args.robust else np.mean
    header = f"{'config':14s} {('final return ' + center_label):30s} {'AUC':10s} {'std':7s}"
    if args.threshold is not None:
        header += f" {'steps@thr (median, %solved)':28s}"
    print(header)
    finals = {}
    for name, (xs, ys) in runs.items():
        fr = final_return_per_seed(ys)
        finals[name] = fr
        auc = auc_per_seed(xs, ys)
        line = f"{name:14s} {fmt_center(fr, args.robust):30s} {auc_center(auc):8.1f}  {np.std(fr):6.1f}"
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
        with warnings.catch_warnings():
            # Saturated configs are near-identical -> harmless precision-loss
            # notices from the t-test; the resulting p-values are still valid.
            warnings.simplefilter("ignore")
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
