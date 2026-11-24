"""First reinforcement-learning bidder for the weekly mFRR auctions (1 MW).

The agent plays one auction after the other:
    it sees     how the last price moved (fell a lot ... rose a lot)
    it chooses  a factor; its bid is factor x last known price
    it earns    the factor if it wins (bid < marginal price), else 0
                (revenue / last price, so cheap and expensive years weigh the same)
It keeps Q[state, factor] = expected reward and moves it towards what it earned.

Its bid changes neither the marginal price nor the next state, so there is no
future reward to plan for: this is a contextual bandit, the simplest form of RL.

Two ways to learn (`feedback`):
    "outcome"    only from its own win or loss
    "published"  from the published marginal price, which also tells what every
                 other factor would have earned

It explores at random while it practises on the learning years, replayed
`passes` times. Afterwards it always plays its best factor and keeps learning
from each published result. With "published" and plain averages nothing is
random: every factor is updated every week.

Defaults (3 states, plain averages) were chosen by walk-forward on 2021-2024.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.bidders import LAST_LEARN_YEAR, _as_bids

MOVE_EDGES = {1: (), 3: (-0.05, 0.05), 5: (-0.15, -0.05, 0.05, 0.15)}


class RLBidder:
    """Set by fit: q_ ({direction: Q table after the learning years}), bids_ (later auctions)."""

    def __init__(self, feedback: str = "outcome", n_states: int = 3, step: float | None = None,
                 epsilon: float = 0.2, passes: int = 50, seed: int = 0,
                 last_learn_year: int = LAST_LEARN_YEAR, factors=None):
        if feedback not in ("outcome", "published"):
            raise ValueError("feedback must be 'outcome' or 'published'")
        self.name = f"rl_{feedback}"
        self.feedback = feedback
        self.n_states = n_states          # 1 = the agent ignores how the price moved
        self.step = step                  # None = plain average of all past rewards
        self.epsilon = epsilon            # share of random bids while practising
        self.passes = passes
        self.seed = seed
        self.last_learn_year = last_learn_year
        self.factors = np.round(np.arange(0.50, 1.0501, 0.05), 2) if factors is None else np.asarray(factors)

    def fit(self, weekly: pd.DataFrame) -> RLBidder:
        """Practise on the learning years, then play every later auction in order.

        Later auctions are only used once published: the bid for an auction comes
        from results with avail <= its gate closure.
        """
        table = weekly.sort_values("gc")
        before = table.groupby("direction")["last_price"].shift(1)
        table = table.assign(move=np.log(table["last_price"] / before),
                             ratio=table["marginal"] / table["last_price"])
        table = table[table["move"].notna()]
        self.q_, self.bids_ = {}, {}
        for i, (direction, g) in enumerate(table.groupby("direction")):
            learn, later = g[g["year"] <= self.last_learn_year], g[g["year"] > self.last_learn_year]
            if learn.empty:
                raise ValueError("no auction to learn from")
            rng = np.random.default_rng([self.seed, i])
            q = np.zeros((self.n_states, len(self.factors)))
            count = np.zeros_like(q)
            self._practise(q, count, learn, rng)
            self.q_[direction] = q.copy()
            self._play(q, count, later)
        return self

    def bid(self, auctions: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "bids_"):
            raise RuntimeError("fit() was not called")
        missing = set(auctions["auction_id"]) - set(self.bids_)
        if missing:
            raise ValueError(f"{len(missing)} auctions were not played after the learning years")
        return _as_bids(auctions, auctions["auction_id"].map(self.bids_).astype(float))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _state(self, move) -> np.ndarray:
        return np.digitize(move, MOVE_EDGES[self.n_states])

    def _practise(self, q, count, learn, rng) -> None:
        states, ratios = self._state(learn["move"].to_numpy()), learn["ratio"].to_numpy()
        for _ in range(self.passes):
            for s, ratio in zip(states, ratios):
                explore = rng.random() < self.epsilon
                a = rng.integers(len(self.factors)) if explore else int(np.argmax(q[s]))
                self._update(q, count, s, a, ratio)

    def _play(self, q, count, later) -> None:
        pending = []                                     # (avail, state, action, ratio)
        for row in later.itertuples():
            for p in [p for p in pending if p[0] <= row.gc]:
                self._update(q, count, *p[1:])
            pending = [p for p in pending if p[0] > row.gc]
            s = int(self._state(row.move))
            a = int(np.argmax(q[s]))
            self.bids_[row.auction_id] = float(self.factors[a] * row.last_price)
            pending.append((row.avail, s, a, row.ratio))

    def _update(self, q, count, s, a, ratio) -> None:
        tried = np.arange(len(self.factors)) if self.feedback == "published" else np.array([a])
        reward = np.where(self.factors[tried] < ratio, self.factors[tried], 0.0)
        count[s, tried] += 1
        step = 1 / count[s, tried] if self.step is None else self.step
        q[s, tried] += step * (reward - q[s, tried])
