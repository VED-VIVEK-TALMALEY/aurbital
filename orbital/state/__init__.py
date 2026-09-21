"""ORBITAL State — persistent world state and observation model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from orbital.ontology import Entity, WorldSchema


@dataclass
class Observation:
    """A noisy, partial measurement with provenance."""
    entity_id: str
    dim: str
    value: float
    t: int
    source: str = "unknown"
    sigma: float = 0.05


class WorldState:
    """Persistent state store: entity -> dim -> full time series.

    The world state *persists* across timesteps — this is the core difference
    from stateless image -> prediction pipelines.
    """

    def __init__(self, schema: WorldSchema):
        self.schema = schema
        self.history: Dict[str, Dict[str, List[float]]] = {}
        for e in schema.entities.values():
            self.history[e.id] = {dim: [v] for dim, v in e.state.items()}

    def current(self, entity_id: str) -> Dict[str, float]:
        return {dim: vals[-1] for dim, vals in self.history[entity_id].items()}

    def series(self, entity_id: str, dim: str) -> np.ndarray:
        return np.asarray(self.history[entity_id].get(dim, []), dtype=float)

    def commit(self, entity_id: str, new_state: Dict[str, float]) -> None:
        """Append a new state (registers new dims on first sight)."""
        e = self.schema.entities[entity_id]
        for dim, val in new_state.items():
            if dim not in self.history[entity_id]:
                self.history[entity_id][dim] = [val]  # backfill first value
        e.state.update(new_state)
        e.clamp_state()
        for dim in self.history[entity_id]:
            self.history[entity_id][dim].append(e.state[dim])

    def assimilate(self, obs: Observation) -> float:
        """Simple scalar Kalman-style update toward an observation.

        Returns the innovation (|measurement − prior|), useful for
        contradiction diagnostics.
        """
        prior = self.current(obs.entity_id).get(obs.dim)
        if prior is None:
            return 0.0
        innovation = abs(obs.value - prior)
        k = obs.sigma**2 / (obs.sigma**2 + 0.02)  # 0.02 = prior process var
        posterior = prior + k * (obs.value - prior)
        self.schema.entities[obs.entity_id].state[obs.dim] = posterior
        return innovation

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {eid: self.current(eid) for eid in self.history}


class StateEstimator:
    """Filters observations into the persistent state (assimilation loop)."""

    def __init__(self, world_state: WorldState):
        self.ws = world_state
        self.innovation_log: List[Observation] = []

    def step(self, observations: List[Observation]) -> List[float]:
        innovations = []
        for o in observations:
            innovations.append(self.ws.assimilate(o))
            self.innovation_log.append(o)
        return innovations
