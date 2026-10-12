"""Offline experiment: does the RL bandit help a learner improve faster?

We cannot recruit hundreds of real students in one day, so we simulate them
(a standard way to test tutoring policies before a user study).

Simulated learner
  * 8 topics; each topic has a hidden probability p of answering correctly.
  * After a WRONG answer the learner reads the explanation and improves:
        p <- p + 0.15 * (1 - p)      (after a correct answer: p <- p + 0.03 * (1 - p))
  * Two learner types (scenarios):
      A "mixed"     : every topic p ~ Uniform(0.30, 0.90)
      B "two-weak"  : 2 weak topics p ~ U(0.25, 0.45), 6 strong topics p ~ U(0.75, 0.95)
                      (a typical reviewer: good overall, a couple of blind spots)

Policies (60-question session, 500 simulated learners, identical learners
for every policy = paired comparison):
  Random, Round-robin, Epsilon-greedy (eps = 0.2), Thompson sampling
  gamma = 1.0, Thompson sampling gamma = 0.98 (the app), and an Oracle that
  always knows the weakest topic (upper bound, impossible in practice).

Metrics
  final average mastery, final weakest-topic mastery, and the share of
  questions spent on the learner's two weakest topics.

Usage:  python scripts/simulate_bandit.py
Writes reports/bandit_simulation.json and reports/bandit_simulation.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from driveready import config  # noqa: E402
from driveready.bandit import EpsilonGreedyPolicy, OraclePolicy, RandomPolicy, RoundRobinPolicy, TopicBandit  # noqa: E402

K = len(config.QUIZ_TOPICS)
QUESTIONS = 60
STUDENTS = 500
LEARN_WRONG, LEARN_RIGHT = 0.15, 0.03
POLICIES = ["Random", "Round-robin", "Epsilon-greedy (ε=0.2)", "Thompson (γ=1.0)",
            "Thompson (γ=0.98, app)", "Oracle (upper bound)"]


def make_policy(name: str, seed: int, true_p: np.ndarray):
    return {
        "Random": lambda: RandomPolicy(K, seed),
        "Round-robin": lambda: RoundRobinPolicy(K),
        "Epsilon-greedy (ε=0.2)": lambda: EpsilonGreedyPolicy(K, 0.2, seed),
        "Thompson (γ=1.0)": lambda: TopicBandit(gamma=1.0, seed=seed),
        "Thompson (γ=0.98, app)": lambda: TopicBandit(gamma=0.98, seed=seed),
        "Oracle (upper bound)": lambda: OraclePolicy(true_p),
    }[name]()


def initial_skills(scenario: str, rng: np.random.Generator) -> np.ndarray:
    if scenario == "A":
        return rng.uniform(0.30, 0.90, size=K)
    p = np.concatenate([rng.uniform(0.25, 0.45, 2), rng.uniform(0.75, 0.95, K - 2)])
    rng.shuffle(p)
    return p


def run_student(policy_name: str, scenario: str, seed: int) -> dict:
    p = initial_skills(scenario, np.random.default_rng(seed))
    weak = set(np.argsort(p)[:2].tolist())
    answer_rng = np.random.default_rng(seed + 10_000)
    policy = make_policy(policy_name, seed + 20_000, p)
    curve, on_weak = [p.mean()], 0
    for _ in range(QUESTIONS):
        arm = policy.select()
        on_weak += arm in weak
        correct = bool(answer_rng.random() < p[arm])
        policy.update(arm, correct)
        p[arm] += (LEARN_RIGHT if correct else LEARN_WRONG) * (1 - p[arm])
        curve.append(p.mean())
    return {"curve": np.array(curve), "final_min": p.min(), "share_weak": on_weak / QUESTIONS}


def summarise(runs: list[dict]) -> dict:
    final = np.array([r["curve"][-1] for r in runs])
    start = np.array([r["curve"][0] for r in runs])
    return {
        "final_mean_mastery": round(float(final.mean()), 4),
        "ci95": round(float(1.96 * final.std(ddof=1) / np.sqrt(len(runs))), 4),
        "mean_gain": round(float((final - start).mean()), 4),
        "final_weakest_topic": round(float(np.mean([r["final_min"] for r in runs])), 4),
        "share_on_two_weakest": round(float(np.mean([r["share_weak"] for r in runs])), 3),
    }


def main() -> None:
    out = {"setup": {"topics": K, "questions_per_session": QUESTIONS, "learners": STUDENTS,
                     "learning_rate_wrong": LEARN_WRONG, "learning_rate_right": LEARN_RIGHT,
                     "scenario_A": "all topics p ~ U(0.30, 0.90)",
                     "scenario_B": "2 weak topics p ~ U(0.25, 0.45), 6 strong p ~ U(0.75, 0.95)"},
           "results": {}}
    curves = {}
    for scen in ("A", "B"):
        out["results"][scen] = {}
        for name in POLICIES:
            runs = [run_student(name, scen, s) for s in range(STUDENTS)]
            out["results"][scen][name] = summarise(runs)
            curves[(scen, name)] = np.mean([r["curve"] for r in runs], axis=0)
        rnd = out["results"][scen]["Random"]["mean_gain"]
        for name in POLICIES:
            r = out["results"][scen][name]
            r["gain_vs_random_pct"] = round(100 * (r["mean_gain"] / rnd - 1), 1)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "bandit_simulation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"Random": ("#9E9E9E", "--"), "Round-robin": ("#6D6D6D", ":"),
              "Epsilon-greedy (ε=0.2)": ("#E39B2D", "-."), "Thompson (γ=1.0)": ("#7FA7D4", "-"),
              "Thompson (γ=0.98, app)": ("#1F4E79", "-"), "Oracle (upper bound)": ("#2E7D32", (0, (1, 1)))}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), dpi=200, sharey=True)
    for ax, scen, title in zip(axes, ("A", "B"), ("A: mixed skills", "B: two weak topics")):
        for name in POLICIES:
            c, ls = styles[name]
            ax.plot(curves[(scen, name)], color=c, ls=ls, lw=2.2 if "app" in name else 1.3, label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Questions answered", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Average topic mastery", fontsize=8)
    axes[1].legend(fontsize=6.3, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "bandit_simulation.png")
    print("saved reports/bandit_simulation.png")


if __name__ == "__main__":
    main()
