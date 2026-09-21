"""ORBITAL Simulation — forward rollouts, counterfactuals, constraints."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from orbital.ontology import Event, WorldSchema
from orbital.models import EnsembleWorldModel, TransitionModel, DIMS
from orbital.state import WorldState

BOUNDS = {
    "ndvi": (0.0, 1.0), "soil_moisture": (0.0, 1.0),
    "temperature": (-30.0, 55.0), "stress": (0.0, 1.0),
}


@dataclass
class RolloutResult:
    states: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    epistemic: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    violations: int = 0
    steps: int = 0


def check_constraints(state: Dict[str, Dict[str, float]]) -> int:
    v = 0
    for eid, s in state.items():
        for d, (lo, hi) in BOUNDS.items():
            if d in s and not (lo - 1e-6 <= s[d] <= hi + 1e-6):
                v += 1
    return v


class Simulator:
    """Rolls the world model forward; optionally under interventions."""

    def __init__(self, schema: WorldSchema, ensemble: EnsembleWorldModel):
        self.schema = schema
        self.ensemble = ensemble

    def rollout(self, world_state: WorldState, horizon: int,
                season: float = 0.0, season_step: float = 1/52,
                interventions: Optional[List[Event]] = None) -> RolloutResult:
        """Roll every entity forward `horizon` steps from the current state.

        `interventions` implements the do-operator: applied on the rollout
        copy only — the observed history is untouched.
        """
        schema = copy.deepcopy(self.schema)
        # reset entity states to current observed values
        for eid, s in world_state.snapshot().items():
            schema.entities[eid].state.update(s)

        result = RolloutResult()
        for k in range(horizon):
            if interventions:
                for ev in interventions:
                    if ev.t == k:
                        from orbital.models import apply_events
                        apply_events(schema, [ev])
            # season at the time PREDICTING step k+1 from step k:
            # k=0 means "the transition out of the last observed step"
            mean, std = self.ensemble.predict_step(
                (season + (k + 1) * season_step) % 1.0)
            result.states.append(mean)
            result.epistemic.append(std)
            result.violations += check_constraints(mean)
            # feed prediction back in as new current state (clamped to bounds)
            for eid, s in mean.items():
                clamped = dict(s)
                for d, (lo, hi) in BOUNDS.items():
                    if d in clamped:
                        clamped[d] = float(np.clip(clamped[d], lo, hi))
                schema.entities[eid].state.update(clamped)
        result.steps = horizon
        return result


def counterfactual_effect(observed: RolloutResult,
                          intervened: RolloutResult,
                          entity_id: str, dim: str) -> np.ndarray:
    """CFE_k = E[z(t+k)|do(e')] − z_obs(t+k) for each horizon step."""
    a = np.array([s[entity_id][dim] for s in observed.states])
    b = np.array([s[entity_id][dim] for s in intervened.states])
    return b - a
