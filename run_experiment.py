import json
import multiprocessing as mp
import os

import numpy as np

import train_ppo


def create_experiment_dir(root_dir, exp_name):
    path = os.path.join(root_dir, exp_name)
    os.makedirs(path, exist_ok=True)

    return path


def main(config_file: str, root_dir: str, n_runs: int = 1, num_processes=4, init_seed=98):
    """Run experiments in parallel."""
    np.random.seed(init_seed)
    seeds = np.random.randint(0, 100000000, n_runs).tolist()

    with open(config_file, "r") as f:
        sweeps = json.load(f)

    for sweep in sweeps:
        sweep["out_dir"] = create_experiment_dir(
            root_dir=root_dir, exp_name=sweep["out_dir"]
        )

    configs = []
    for seed in seeds:
        for config in sweeps:
            if os.path.isfile(os.path.join(config["out_dir"], f"{seed}.txt")):
                print(f"Data for run {seed} already existing. Skipping it...")
                continue
            _config = config.copy()
            _config["seed"] = seed
            configs.append(_config)

    print(f"Running {n_runs * len(sweeps)} experiments in total.")
    print(f"Running {min(num_processes, len(configs))} experiments in parallel.")

    for config in configs:
        print(config)

    pool = mp.Pool(processes=num_processes)
    results = [
        pool.apply_async(train_ppo.run, kwds=dict(**dict(config))) for config in configs
    ]
    results = [p.get() for p in results]

    for i in range(len(results)):
        avg_time = np.mean(results[i])
        configs[i]["avg_time"] = avg_time
        with open(os.path.join(configs[i]["out_dir"], "stats.txt"), "w") as f:
            json.dump(configs[i], f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", default=None, type=str, help="Config file")
    parser.add_argument("--n-runs", type=int, default=30, help="Number of runs")
    parser.add_argument("--n-proc", type=int, default=4, help="Number of processes")
    parser.add_argument(
        "--main-seed",
        dest="main_seed",
        type=int,
        default=98,
        help="Main seed for creatings sequence of seeded runs.",
    )
    parser.add_argument(
        "--root-dir", type=str, default="results", help="Root directory"
    )

    args = parser.parse_args()

    main(args.config_file, args.root_dir, args.n_runs, num_processes=args.n_proc, init_seed=args.main_seed)
