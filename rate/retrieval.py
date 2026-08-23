"""Retrieval of a similar historical event.

Every event is summarised, once per second of elapsed time, by the vector of
peak ground accelerations observed so far at each station - its "spatial
intensity signature" - normalised to unit length.  Similarity between two
events is then the cosine between two such vectors, which FAISS resolves with
an inner-product index (one index per second, since the current event can only
be compared against equally long observations).

The pool is the training split only, so a test event can never retrieve a
future event.  A *training* event, however, is itself in the pool: see
``RetrievalConfig.exclude_self``.
"""

from __future__ import annotations

import faiss
import numpy as np

from .config import RetrievalConfig


class RetrievalIndex:
    def __init__(self, database_path: str, pool: int, topk: int = 1, exclude_self: bool = False):
        self.signatures = np.load(database_path)
        if self.signatures.ndim != 3:
            raise ValueError(
                f"{database_path}: expected (events, stations, time steps), got {self.signatures.shape}"
            )
        if pool > len(self.signatures):
            raise ValueError(
                f"{database_path} holds {len(self.signatures)} events, "
                f"fewer than the {pool} events of the retrieval pool"
            )
        self.pool = pool
        self.topk = topk
        self.exclude_self = exclude_self
        self.time_steps = self.signatures.shape[-1]
        self.indexes = []
        for step in range(self.time_steps):
            index = faiss.IndexFlatIP(self.signatures.shape[-2])
            index.add(np.ascontiguousarray(self.signatures[: self.pool, :, step]))
            self.indexes.append(index)

    @classmethod
    def build(cls, config: RetrievalConfig, pool: int) -> "RetrievalIndex | None":
        if not config.enabled:
            return None
        return cls(config.database, pool=pool, topk=config.topk, exclude_self=config.exclude_self)

    def time_step(self, cutout: int, sampling_rate: int) -> int:
        """Which one-second signature to query for a cutout given in samples."""
        return min(cutout // sampling_rate, self.time_steps - 1)

    def neighbours(self, event: int, step: int, k: int) -> np.ndarray:
        """The ``k`` most similar pool events, best first (-1 pads a short result)."""
        query = np.ascontiguousarray(self.signatures[event, :, step][None, :])
        wanted = k + 1 if self.exclude_self else k
        found = self.indexes[step].search(query, wanted)[1][0]
        if self.exclude_self:
            found = found[found != event][:k]
            found = np.pad(found, (0, k - len(found)), constant_values=-1)
        return found

    def pick(self, event: int, step: int) -> int:
        """One neighbour, drawn uniformly from the top-k. -1 if there is none."""
        candidates = self.neighbours(event, step, self.topk)
        if self.topk > 1:
            np.random.shuffle(candidates)
        return int(candidates[0])
