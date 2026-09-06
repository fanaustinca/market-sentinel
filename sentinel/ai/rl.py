"""A reinforcement-learning trader, and an honest account of what it can learn.

Why this is a contextual bandit wearing a costume
--------------------------------------------------
Reinforcement learning earns its keep when actions change the world -- a move
alters the state, which alters the next choice, so the agent must plan. A trader
holding liquid ETFs does not move the market. Tomorrow's prices are identical
whether the agent went long or stayed flat, so the environment's dynamics are
entirely exogenous.

One thing does connect the steps: turnover costs money, so the position held
yesterday changes what today's choice is worth. That is genuine sequential
structure and it is why the previous position is part of the state here. But it
is *all* of the sequential structure there is, and it concerns costs rather than
returns. Everything else in this environment is a bandit: observe features,
choose a size, receive a reward drawn from a distribution the agent cannot
influence.

That matters because it says what RL can possibly discover. It cannot learn to
move the market, time an opponent, or plan a sequence -- there is nobody to plan
against. It can learn a mapping from features to position size, which is the same
object the supervised model learns, arrived at by a slower and noisier route.

And `experiments/perfect_timing.py` already bounds that mapping. A supervised
model handed the *perfect answers* for 531,728 stock-days reproduced them on
unseen data 50.8% of the time, against 52.6% for always guessing up. An RL agent
has strictly less information: it must discover those answers from noisy rewards
rather than being told them. It cannot exceed its own answer key.

So this is built to be measured, not to be believed. The single thing worth
watching is whether the agent discovers *risk* control -- sizing down when
volatility is high -- because that is the one policy this project has repeatedly
found to be learnable. If it converges on volatility targeting, that is a real
result about what is in the data, and it is also not more money.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Position sizes the agent may choose. Long-only and unlevered, matching the
#: policy every other strategy here obeys.
ACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class TrainingLog:
    """What happened during fitting, kept so failure is diagnosable."""

    rewards: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    mean_positions: list[float] = field(default_factory=list)


class PolicyGradientTrader:
    """REINFORCE with a learned baseline, over a small feed-forward policy.

    Args:
        features: number of input columns, plus one for the previous position.
        hidden: width of the single hidden layer. Deliberately small -- the
            supervised experiment showed capacity buys memorisation here, not
            generalisation, and a larger network would only memorise faster.
        cost_bps: charged on every unit of position changed, so the agent feels
            turnover the way the backtest engine does.
        entropy_bonus: keeps the policy from collapsing onto one action early.
            Without it REINFORCE on a near-zero-signal problem reliably converges
            to a constant policy in the first few epochs and learns nothing after.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        learning_rate: float = 1e-3,
        cost_bps: float = 5.0,
        entropy_bonus: float = 0.01,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        self.n_features = n_features
        self.cost = cost_bps / 10_000.0
        self.entropy_bonus = float(entropy_bonus)

        # +1 input for the previous position, which is the only genuine state
        # variable in this environment.
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(n_features + 1, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, len(ACTIONS)),
        ).to(self.device)
        self.baseline = torch.nn.Sequential(
            torch.nn.Linear(n_features + 1, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        ).to(self.device)

        self.optimiser = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.baseline.parameters()), lr=learning_rate
        )
        self.actions = torch.tensor(ACTIONS, dtype=torch.float32, device=self.device)
        self.log = TrainingLog()

    def _forward(self, states):
        return self.policy(states), self.baseline(states).squeeze(-1)

    def fit(self, features: np.ndarray, returns: np.ndarray, epochs: int = 30,
            batch: int = 4096) -> TrainingLog:
        """Train on (features, next-period return) pairs drawn as one-step episodes.

        Sampling minibatches of independent steps rather than replaying whole
        price paths is deliberate: the environment has no state transition the
        agent controls beyond its own position, so a contiguous episode carries no
        extra information and merely correlates the gradient.
        """
        torch = self.torch
        X = torch.tensor(features, dtype=torch.float32, device=self.device)
        R = torch.tensor(returns, dtype=torch.float32, device=self.device)
        n = len(X)

        for _ in range(epochs):
            order = torch.randperm(n, device=self.device)
            epoch_reward, epoch_entropy, epoch_position = [], [], []

            for start in range(0, n, batch):
                index = order[start : start + batch]
                previous = torch.rand(len(index), 1, device=self.device)
                states = torch.cat([X[index], previous], dim=1)

                logits, value = self._forward(states)
                distribution = torch.distributions.Categorical(logits=logits)
                choice = distribution.sample()
                position = self.actions[choice]

                turnover = (position - previous.squeeze(-1)).abs()
                reward = position * R[index] - turnover * self.cost

                advantage = (reward - value).detach()
                policy_loss = -(distribution.log_prob(choice) * advantage).mean()
                value_loss = torch.nn.functional.mse_loss(value, reward)
                entropy = distribution.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - self.entropy_bonus * entropy

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                # Detached before logging: these are diagnostics, and keeping
                # them attached to the graph holds the whole batch alive.
                epoch_reward.append(float(reward.mean().detach()))
                epoch_entropy.append(float(entropy.detach()))
                epoch_position.append(float(position.mean().detach()))

            self.log.rewards.append(float(np.mean(epoch_reward)))
            self.log.entropies.append(float(np.mean(epoch_entropy)))
            self.log.mean_positions.append(float(np.mean(epoch_position)))
        return self.log

    def act(self, features: np.ndarray, greedy: bool = True) -> np.ndarray:
        """Positions for each row, rolling the previous choice forward as state."""
        torch = self.torch
        X = torch.tensor(features, dtype=torch.float32, device=self.device)
        out = np.zeros(len(X))
        previous = 0.0
        with torch.no_grad():
            # Stepped rather than vectorised because the previous position is an
            # input: vectorising would require knowing it in advance, which is the
            # same off-by-one that makes a backtest read tomorrow's newspaper.
            for i in range(len(X)):
                state = torch.cat([X[i], torch.tensor([previous], device=self.device)]).unsqueeze(0)
                logits = self.policy(state)
                choice = int(logits.argmax(dim=-1)) if greedy else int(
                    torch.distributions.Categorical(logits=logits).sample()
                )
                previous = float(ACTIONS[choice])
                out[i] = previous
        return out
