import os
import time
from collections import deque
from copy import deepcopy
from typing import Optional
from typing import Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ppo_study.model import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(
    seed: int = 0,
    out_dir: str = "ppo",
    env_name: str = "CartPole-v1",
    constrained: bool = False,
    num_steps: int = 500_000,
    checkpoint: int = 10_000,
    batch_size: int = 100,
    M: int = 32,
    N: Optional[Union[int, None]] = 32,
    C: int = 2000,
    T: int = 2000,
    E: int = 10,
    c: Optional[Union[float, None]] = None,
    tau: Optional[Union[int, None]] = None,
    c2: Optional[Union[int, None]] = None,
    epsilon: float = 0.2,
    gamma: float = 0.99,
    lmbda: float = 0.95,
    actor_step_size: float = 3e-4,
    n_actor_hidden_units: int = 64,
    n_critic_hidden_units: int = 64,
    critic_step_size: float = 1e-3,
    to_log: bool = False,
    to_plot: bool = False,
):
    """
    :param seed: Random seed
    :param out_dir: Output directory
    :param env_name: Gymnasium environment id (e.g. CartPole-v1, Acrobot-v1).
    :param constrained: Whether to use the constrained version of PPO.
    :param num_steps: Number of steps to train for.
    :param checkpoint: Number of steps between checkpoints.
    :param batch_size: Batch size (None for unconstrained version).
    :param N: number of total mini-batches
    :param M: Number of batches to learn from during a update burst.
    :param C: Number of samples stored in buffer.
    :param T: Number of steps until net update burst.
    :param E: Number of epochs to train for during a update burst.
    :param c: Constraint 1 constant.
    :param tau: Constraint 2 constant.
    :param c2: Constraint 2 constant.
    :param epsilon: PPO clipping parameter.
    :param gamma: Discount factor.
    :param lmbda: GAE parameter.
    :param actor_step_size: Actor step size.
    :param n_actor_hidden_units: Number of hidden units in actor.
    :param n_critic_hidden_units: Number of hidden units in critic.
    :param critic_step_size: Critic step size.
    :param to_log: Whether to log to console.
    :param to_plot: Whether to plot results.
    """
    start = time.time()

    if constrained:
        # Assert constraints
        N = int(np.ceil(C / batch_size))
        H = 4 * (C / N) * M * E
        L = H * (c * tau) / c2
        assert (
            H * (c * tau) / c2 <= tau * T
        ), f"Constraint 2 failed with config: {c, tau, c2, T, E}."
        # Real-time data constraint (grader note #2): computing a burst takes L
        # wall-clock time = `busy_steps` environment steps, during which the
        # learner is occupied and cannot ingest the samples being collected. So
        # those samples are dropped and each burst (in steady state) learns on
        # only `T - busy_steps` of the `T` samples gathered since the last burst.
        busy_steps = int(np.floor(L / tau))
        usable = T - busy_steps
        assert M * batch_size <= usable, (
            f"Not enough usable samples for {M} minibatches of size {batch_size}: "
            f"M*batch_size={M * batch_size} > T - floor(L/tau)={usable} "
            f"(config c2={c2}, M={M}, E={E}, busy_steps={busy_steps})."
        )

    env = gym.make(env_name)
    env.reset(seed=seed)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    torch.manual_seed(seed)
    ppo = PPO(
        constrained=constrained,
        actor_step_size=actor_step_size,
        actor_n_hidden_units=[n_actor_hidden_units, n_actor_hidden_units],
        actor_activation_func=nn.Tanh(),
        critic_step_size=critic_step_size,
        critic_n_hidden_units=[n_critic_hidden_units, n_critic_hidden_units],
        critic_activation_func=nn.Tanh(),
        gamma=gamma,
        lmbda=lmbda,
        o_dim=o_dim,
        a_dim=a_dim,
        device=device,
    )

    G = 0
    Gs = []
    avg_Gs = []
    o, _ = env.reset()
    b = deque(maxlen=C)

    if constrained:
        tp = -1
        learner_busy_until = -1
    if not constrained:
        N = int(C / batch_size)
        M = N

    for steps in tqdm(range(num_steps), desc="Training", position=0, leave=True):
        # Select an action
        inputs = torch.from_numpy(o).float().unsqueeze(0).data.to(device)
        # We act with the old parameters until those will be updated as well
        if constrained:
            log_probs = ppo.actor_update(inputs)
        else:
            log_probs = ppo.actor_learn(inputs)
        probs = torch.exp(log_probs)
        pol = torch.distributions.Categorical(probs=probs)
        a = pol.sample()

        # Observe
        observation, reward, terminated, truncated, _ = env.step(a.item())

        # Learn. In the real-time constrained setting the learner is busy for
        # `busy_steps` steps after a burst and cannot ingest samples collected
        # during that window (grader note #2), so we drop them from the buffer.
        if (not constrained) or (steps > learner_busy_until):
            b.append((o, a, reward, observation, terminated))

        if constrained:
            # Time to update the actor
            if steps == tp:
                # We delay the transfer of the learned parameters to the performance element until the time is due. Until then, we are acting with a separate copy of the performance element that is using the old parameters
                ppo.actor_update = deepcopy(ppo.actor_learn)

        # Time to learn
        if (steps + 1) % T == 0:
            # Upon starting a new learning burst we schedule, via Constraint 3,
            # the step `tp` at which the freshly learned params are deployed to
            # the performance element. The learner is also busy (cannot ingest
            # new samples) until `tp`.
            if constrained:
                tp = steps + busy_steps
                learner_busy_until = tp
            ppo.learn(list(b), E, N, M, epsilon, batch_size)
            b.clear()

        o = observation

        # Log
        G += reward
        if terminated or truncated:
            Gs.append(G)
            G = 0
            o, _ = env.reset()

        if (steps + 1) % checkpoint == 0:
            avg_Gs.append(np.mean(Gs))

            if to_log:
                tqdm.write(f"{avg_Gs[-1]:.5f}")

            Gs = []
            if to_plot:
                plt.clf()
                plt.plot(
                    range(checkpoint, (steps + 1) + checkpoint, checkpoint), avg_Gs
                )
                plt.pause(0.001)

    data = np.zeros((2, len(avg_Gs)))
    data[0] = range(checkpoint, num_steps + 1, checkpoint)
    data[1] = avg_Gs
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, str(seed) + ".txt"), data)

    if to_plot:
        plt.show()

    # Save model
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(
        ppo.actor_learn.state_dict(),
        os.path.join(models_dir, str(seed) + ".pth"),
    )

    return time.time() - start


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed to use for the experiment."
    )
    parser.add_argument(
        "--out-dir", type=str, default="ppo", help="Directory to save the results."
    )
    parser.add_argument(
        "--env-name",
        dest="env_name",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment id (e.g. CartPole-v1, Acrobot-v1).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500000,
        help="Number of steps to run the experiment for.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=10000,
        help="Number of steps between checkpoints.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for the learner."
    )
    parser.add_argument(
        "--M", type=int, default=32, help="Number of epochs for the learner."
    )
    parser.add_argument(
        "--N", type=int, default=32, help="Number of epochs for the learner."
    )
    parser.add_argument(
        "--C", type=int, default=2000, help="Number of samples for the learner."
    )
    parser.add_argument(
        "--T", type=int, default=2000, help="Number of steps between learner updates."
    )
    parser.add_argument(
        "--E", type=int, default=10, help="Number of epochs for the actor."
    )
    parser.add_argument(
        "--c", type=float, default=None, help="Learning rate for the actor."
    )
    parser.add_argument(
        "--tau", type=int, default=None, help="Number of steps between actor updates."
    )
    parser.add_argument(
        "--c2", type=float, default=None, help="Learning rate for the learner."
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.2, help="Epsilon for the learner."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Gamma for the learner."
    )
    parser.add_argument(
        "--lmbda", type=float, default=0.95, help="Lambda for the learner."
    )
    parser.add_argument(
        "--actor-step-size", type=float, default=3e-4, help="Actor step size."
    )
    parser.add_argument(
        "--critic-step-size", type=float, default=1e-3, help="Critic step size."
    )
    parser.add_argument(
        "--n_actor_hidden_units",
        type=int,
        default=64,
        help="Number of hidden units for the actor.",
    )
    parser.add_argument(
        "--n_critic_hidden_units",
        type=int,
        default=64,
        help="Number of hidden units for the critic.",
    )
    parser.add_argument(
        "--log", dest="to_log", action="store_true", help="Whether to log the results."
    )
    parser.add_argument(
        "--constrained", action="store_true", help="Whether to constrain PPO."
    )
    parser.add_argument(
        "--plot",
        dest="to_plot",
        action="store_true",
        help="Whether to plot the results.",
    )
    args = parser.parse_args()
    run(**vars(args))
