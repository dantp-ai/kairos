from copy import deepcopy

import torch
import torch.nn as nn


class PPO:
    def __init__(
        self,
        o_dim,
        a_dim,
        device,
        actor_step_size=0.005,
        actor_n_hidden_units=[16, 32],
        actor_activation_func=nn.ReLU(),
        critic_step_size=0.005,
        critic_n_hidden_units=[16, 32],
        critic_activation_func=nn.ReLU(),
        critic_out_size=1,
        lmbda=1.0,
        gamma=1.0,
        constrained=False,
    ):
        super().__init__()

        self.device = device
        self.constrained = constrained
        # Actor
        step_size_actor = actor_step_size
        self.actor_learn = nn.Sequential(
            nn.Linear(o_dim, actor_n_hidden_units[0]),
            actor_activation_func,
            nn.Linear(actor_n_hidden_units[0], actor_n_hidden_units[1]),
            actor_activation_func,
            nn.Linear(actor_n_hidden_units[1], a_dim),
            nn.LogSoftmax(dim=-1),
        ).to(device)

        if constrained:
            self.actor_update = deepcopy(self.actor_learn)
        self.optimizer_actor = torch.optim.Adam(
            self.actor_learn.parameters(), lr=step_size_actor
        )

        # Critic
        step_size_critic = critic_step_size
        self.lmbda = lmbda
        self.gamma = gamma
        self.critic = nn.Sequential(
            nn.Linear(o_dim, critic_n_hidden_units[0]),
            critic_activation_func,
            nn.Linear(critic_n_hidden_units[0], critic_n_hidden_units[1]),
            critic_activation_func,
            nn.Linear(critic_n_hidden_units[1], critic_out_size),
        ).to(device)
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=step_size_critic
        )

    def learn(self, b, E, N, M, epsilon, batch_size):
        """
        :param b: buffer
        :param E: number of epochs
        :param N: number of total mini-batches
        :param M: number o mini-batches used for learning (<= N)
        :param epsilon: clipping parameter
        :param batch_size: size of mini-batch
        """
        buffer_size = len(b)
        Glmbda = 0.0
        hlmbda = 0.0

        hlmbdas = []
        Glmbdas = []
        vhats = []
        prev_inputs = []
        actions = []
        action_probs = []

        for transition in reversed(b):
            prev_obs, action, reward, curr_obs, d = transition

            # State-features
            prev_input = torch.from_numpy(prev_obs).float().unsqueeze(0).to(self.device)
            curr_input = torch.from_numpy(curr_obs).float().unsqueeze(0).to(self.device)

            # State-values from state-features
            prev_vhat = self.critic(prev_input)
            curr_vhat = self.critic(curr_input)

            # TD error and lambda-advantage estimate
            td_error = reward + (1 - d) * self.gamma * curr_vhat - prev_vhat
            hlmbda = td_error + (1 - d) * self.gamma * self.lmbda * hlmbda

            # lambda-return
            Glmbda = hlmbda + prev_vhat

            # Preferences and log-probabilities
            # We act with the old parameters until those will be updated as well
            if self.constrained:
                log_probs = self.actor_update(prev_input)
            else:
                log_probs = self.actor_learn(prev_input)
            probs = torch.exp(log_probs)
            pol = torch.distributions.Categorical(probs=probs)
            action_prob = pol.log_prob(action).exp()
            action_prob = action_prob.unsqueeze(0)

            hlmbdas.append(hlmbda)
            Glmbdas.append(Glmbda)
            prev_inputs.append(prev_input)
            actions.append(action)
            action_probs.append(action_prob)

        hlmbdas = torch.cat(hlmbdas)
        Glmbdas = torch.cat(Glmbdas)
        prev_inputs = torch.cat(prev_inputs)
        actions = torch.cat(actions).reshape((-1, 1))
        action_probs = torch.cat(action_probs)

        hlmbdas = (hlmbdas - hlmbdas.mean()) / hlmbdas.std()

        for _ in range(E):

            rand_idcs = torch.randperm(buffer_size)

            hlmbdas = hlmbdas[rand_idcs]
            Glmbdas = Glmbdas[rand_idcs]
            prev_inputs = prev_inputs[rand_idcs]
            actions = actions[rand_idcs]
            action_probs = action_probs[rand_idcs]

            for m in range(M):
                start = m * batch_size
                end = m * batch_size + batch_size
                if M == N and m == M - 1 and end < buffer_size:
                    end = buffer_size

                log_probs = self.actor_learn(prev_inputs[start:end])
                probs = torch.exp(log_probs)
                pol = torch.distributions.Categorical(probs=probs)
                a_probs = pol.log_prob(actions[start:end].squeeze()).exp()
                a_probs = a_probs.reshape((-1, 1))

                term1 = hlmbdas.detach()[start:end] * (
                    a_probs / action_probs.detach()[start:end]
                )
                term2 = (
                    torch.clamp(
                        a_probs / action_probs.detach()[start:end],
                        1 - epsilon,
                        1 + epsilon,
                    )
                    * hlmbdas.detach()[start:end]
                )
                zeta = torch.min(term1, term2)
                lossA = -zeta.mean()

                vhats = self.critic(prev_inputs[start:end])
                lossC = (Glmbdas.detach()[start:end] - vhats).pow(2).mean()

                self.optimizer_critic.zero_grad()
                self.optimizer_actor.zero_grad()

                loss = lossA + lossC
                loss.backward()

                self.optimizer_critic.step()
                self.optimizer_actor.step()
