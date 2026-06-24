#!/usr/bin/env bash
# Generate all learning-curve figures + summary tables for the computational
# -constraint study. Run after `run_experiment.py` has populated results/.
#
#   bash make_figures.sh
#
# CartPole-v1 is "solved" near 475/500; Acrobot-v1 near -100.
set -euo pipefail

run () {  # series_dir  title  threshold
  uv run python analyze.py "results/$1" \
    --title "$2" --threshold "$3" \
    --save "results/figures/$1.png"
  echo
}

run seriesA_cartpole "Series A (compute scaled with c2) - CartPole-v1"  475
run seriesA_acrobot  "Series A (compute scaled with c2) - Acrobot-v1"  -100
run seriesB_cartpole "Series B (epochs vs minibatches) - CartPole-v1"   475
run seriesB_acrobot  "Series B (epochs vs minibatches) - Acrobot-v1"   -100
