import os

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
        }
    )
except Exception as exc:
    print(exc)
    print("Failed to set rc text usetex to True.")
    print("This is fine if you are not using LaTeX.")


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data_run(run_dir: str):
    """Load data from a run."""
    data = []
    for root, _, files in os.walk(run_dir):
        for file in files:
            if file.endswith(".txt") and not file.startswith("stats"):
                with open(os.path.join(root, file), "r") as f:
                    # read txt file to numpy array
                    data_run = np.loadtxt(f)
                    xs = data_run[0]
                    data.append(data_run[1])
    return xs, np.vstack(data)


def plot_learning_curve(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    save_path: str = None,
    show: bool = True,
):
    """Plot learning curve."""
    # plot each row of data as a red line
    for i in range(ys.shape[0]):
        ax.plot(xs, ys[i], color="red", alpha=0.1)
    # plot the mean of data as a blue line
    ax.plot(xs, np.mean(ys, axis=0), color="red")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    # rotate y-axis label to be vertical
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)
    ax.legend([label])
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


# plot a grid of learning curves by using the above function
def plot_learning_curves(
    xs: np.ndarray,
    ys: np.ndarray,
    titles: list,
    xlabels: list,
    ylabels: list,
    labels: list,
    dpi=120,
    save_path: str = None,
):
    """Plot learning curves."""
    create_dir(save_path)

    c = len(ys)
    _, axs = plt.subplots(1, c, figsize=(7 * c, 5), squeeze=False)
    for i in range(c):
        plot_learning_curve(
            axs[0, i],
            xs[i],
            ys[i],
            titles[i],
            xlabels[i] if len(xlabels) == c else None,
            ylabels[i] if len(ylabels) == c else None,
            labels[i],
            show=False,
        )

    for ax in axs.flat:
        ax.label_outer()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "learning_curve.png"), dpi=dpi)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="experiments")
    parser.add_argument("--save-path", type=str, default="")
    args = parser.parse_args()

    xs, ys = load_data_run(args.root_dir)

    xs = np.expand_dims(xs, axis=0)
    ys = np.expand_dims(ys, axis=0)

    plot_learning_curves(
        xs,
        ys,
        ["PPO Learning Curve"] * len(xs),
        [r"Number of steps (x$10^4$)"] * len(xs),
        [r"Average episodic return over last $10^4$ steps"] * len(xs),
        ["PPO unconstrained"] * len(xs),
        save_path=args.save_path,
    )
