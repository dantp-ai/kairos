#!/usr/bin/env bash
# Generate all learning-curve figures + summary tables for the computational
# -constraint study. Run after run_experiment has populated
# experiments/<series>/results/<env>/.
#
#   bash make_figures.sh
#
# CartPole-v1 is "solved" near 475/500; Acrobot-v1 near -100.
set -euo pipefail

run () {  # series  env  title  threshold  [extra flags, e.g. --robust]
  local series="$1" env="$2" title="$3" threshold="$4" extra="${5:-}"
  local dir="experiments/$series/results/$env"
  if [ ! -d "$dir" ] || [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "skip $series/$env (no results yet at $dir)"; echo; return 0
  fi
  uv run python -m ppo_study.analyze "$dir" \
    --title "$title" --threshold "$threshold" $extra \
    --save "experiments/$series/figures/$env.png"
  echo
}

# Acrobot uses --robust (median + IQR): a few seeds diverge on the harder task,
# which would otherwise blow up the mean's bootstrap CI into unreadable bands.
run seriesA cartpole "Series A (compute scaled with c2) - CartPole-v1"  475
run seriesA acrobot  "Series A (compute scaled with c2) - Acrobot-v1"  -100 --robust
run seriesB cartpole "Series B (epochs vs minibatches) - CartPole-v1"   475
run seriesB acrobot  "Series B (epochs vs minibatches) - Acrobot-v1"   -100 --robust
run seriesC cartpole "Series C (fixed work, busy window shrinks) - CartPole-v1"  475
run seriesC acrobot  "Series C (fixed work, busy window shrinks) - Acrobot-v1"  -100 --robust
run seriesD cartpole "Series D (update cadence T=C) - CartPole-v1"  475
run seriesD acrobot  "Series D (update cadence T=C) - Acrobot-v1"  -100 --robust
run seriesE cartpole "Series E (minibatch size at fixed compute) - CartPole-v1"  475
run seriesE acrobot  "Series E (minibatch size at fixed compute) - Acrobot-v1"  -100 --robust
run seriesF cartpole "Series F (environment speed tau) - CartPole-v1"  475
run seriesF acrobot  "Series F (environment speed tau) - Acrobot-v1"  -100 --robust
