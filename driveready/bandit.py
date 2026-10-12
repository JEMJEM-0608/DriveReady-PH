"""Adaptive quiz topic selection with reinforcement learning (multi-armed bandit).

The problem
  The reviewer has 8 quiz topics. A good tutor asks more questions about the
  topics a student keeps getting wrong, but still checks the other topics now
  and then. This is the classic exploration-vs-exploitation trade-off, which
  is exactly what a multi-armed bandit solves.

The RL formulation
  * Agent   : the quiz engine.
  * Actions : choose one of the K topics ("arms") for the next question.
  * Reward  : 1 if the student answers WRONG (we found a weak spot worth
              practising), 0 if correct.
  * Policy  : Thompson sampling. Each topic i keeps a Beta(alpha_i, beta_i)
              belief about its error rate. For every question we sample one
              value from each belief and pick the topic with the largest
              sample. Uncertain topics sometimes win (exploration); topics with
              many recorded mistakes usually win (exploitation).
  * Discount: students improve, so old mistakes should matter less. After
              every update all counts are multiplied by gamma (0.98) and pulled
              back toward the prior ("discounted Thompson sampling"), which lets
              the policy follow a learner whose skills change over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class TopicBandit:
    topics: list[str] = field(default_factory=lambda: list(config.QUIZ_TOPICS))
    gamma: float = 0.98                       # discount factor (1.0 = never forget)
    prior: tuple[float, float] = config.BANDIT_PRIOR
    seed: int | None = None
    alpha: np.ndarray = field(init=False)     # "wrong" evidence per topic
    beta: np.ndarray = field(init=False)      # "correct" evidence per topic
    pulls: np.ndarray = field(init=False)     # how many questions per topic

    def __post_init__(self):
        k = len(self.topics)
        self.alpha = np.full(k, self.prior[0], dtype=float)
        self.beta = np.full(k, self.prior[1], dtype=float)
        self.pulls = np.zeros(k, dtype=int)
        self.rng = np.random.default_rng(self.seed)

    # -- policy ---------------------------------------------------------------
    def select(self) -> int:
        """Thompson sampling: sample each topic's error rate and pick the max."""
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def select_topic(self) -> str:
        return self.topics[self.select()]

    # -- learning -------------------------------------------------------------
    def update(self, arm: int, correct: bool) -> None:
        """Record one answer. reward = 1 for a wrong answer (weak spot found)."""
        a0, b0 = self.prior
        # discount: shrink all evidence toward the prior so recent answers count more
        self.alpha = a0 + self.gamma * (self.alpha - a0)
        self.beta = b0 + self.gamma * (self.beta - b0)
        if correct:
            self.beta[arm] += 1.0
        else:
            self.alpha[arm] += 1.0
        self.pulls[arm] += 1

    def update_topic(self, topic: str, correct: bool) -> None:
        self.update(self.topics.index(topic), correct)

    # -- reporting ------------------------------------------------------------
    def error_rate(self) -> np.ndarray:
        """Posterior mean error rate per topic."""
        return self.alpha / (self.alpha + self.beta)

    def mastery(self) -> dict[str, float]:
        """Estimated probability of answering correctly, per topic (0..1)."""
        return {t: float(1 - e) for t, e in zip(self.topics, self.error_rate())}

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"topics": self.topics, "alpha": self.alpha.tolist(), "beta": self.beta.tolist(),
                "pulls": self.pulls.tolist(), "gamma": self.gamma}

    @classmethod
    def from_dict(cls, d: dict, seed: int | None = None) -> "TopicBandit":
        b = cls(topics=list(d["topics"]), gamma=d.get("gamma", 0.98), seed=seed)
        b.alpha = np.asarray(d["alpha"], dtype=float)
        b.beta = np.asarray(d["beta"], dtype=float)
        b.pulls = np.asarray(d["pulls"], dtype=int)
        return b


class OraclePolicy:
    """Upper bound used only in simulation: always knows the truly weakest topic."""

    def __init__(self, true_p: np.ndarray):
        self.p = true_p            # reference to the simulated learner's real skills

    def select(self) -> int:
        return int(np.argmin(self.p))

    def update(self, arm: int, correct: bool) -> None:
        pass


class RandomPolicy:
    """Baseline: choose a topic uniformly at random (no learning)."""

    def __init__(self, k: int, seed: int | None = None):
        self.k = k
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        return int(self.rng.integers(self.k))

    def update(self, arm: int, correct: bool) -> None:
        pass


class RoundRobinPolicy:
    """Baseline: cycle through the topics in order (what a fixed reviewer does)."""

    def __init__(self, k: int):
        self.k, self.t = k, 0

    def select(self) -> int:
        arm = self.t % self.k
        self.t += 1
        return arm

    def update(self, arm: int, correct: bool) -> None:
        pass


class EpsilonGreedyPolicy:
    """Baseline RL policy: pick the topic with the highest observed error rate,
    but explore a random topic with probability epsilon."""

    def __init__(self, k: int, epsilon: float = 0.2, seed: int | None = None):
        self.k, self.eps = k, epsilon
        self.wrong = np.zeros(k)
        self.n = np.zeros(k)
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        if self.rng.random() < self.eps or self.n.min() == 0:
            untried = np.flatnonzero(self.n == 0)
            return int(self.rng.choice(untried)) if len(untried) else int(self.rng.integers(self.k))
        return int(np.argmax(self.wrong / self.n))

    def update(self, arm: int, correct: bool) -> None:
        self.n[arm] += 1
        self.wrong[arm] += 0 if correct else 1
