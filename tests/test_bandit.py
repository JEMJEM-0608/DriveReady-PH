import numpy as np

from driveready.bandit import RandomPolicy, TopicBandit


def test_bandit_focuses_on_topic_with_mistakes():
    b = TopicBandit(seed=0)
    weak = 2
    for _ in range(30):                     # learner always wrong on topic 2, right elsewhere
        for arm in range(len(b.topics)):
            b.update(arm, correct=(arm != weak))
    picks = [b.select() for _ in range(500)]
    assert np.bincount(picks, minlength=len(b.topics)).argmax() == weak
    assert min(b.mastery(), key=b.mastery().get) == b.topics[weak]


def test_bandit_still_explores():
    b = TopicBandit(seed=1)
    for _ in range(5):
        b.update(0, correct=False)
    picks = {b.select() for _ in range(300)}
    assert len(picks) > 1                   # other topics are still sampled sometimes


def test_discount_forgets_old_mistakes():
    b = TopicBandit(gamma=0.9, seed=0)
    for _ in range(10):
        b.update(0, correct=False)
    early = b.error_rate()[0]
    for _ in range(30):
        b.update(0, correct=True)            # the learner improved
    assert b.error_rate()[0] < 0.2 < early


def test_round_trip_persistence():
    b = TopicBandit(seed=0)
    b.update(3, correct=False)
    c = TopicBandit.from_dict(b.to_dict())
    assert np.allclose(b.alpha, c.alpha) and np.allclose(b.beta, c.beta)
    assert c.pulls[3] == 1


def test_random_policy_range():
    p = RandomPolicy(8, seed=0)
    assert all(0 <= p.select() < 8 for _ in range(100))
