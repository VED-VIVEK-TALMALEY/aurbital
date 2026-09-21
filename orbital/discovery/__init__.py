"""ORBITAL Discovery — active perception via expected information gain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from orbital.state import Observation, WorldState


@dataclass
class CandidateObservation:
    entity_id: str
    dim: str
    cost: float = 1.0
    sigma: float = 0.05
    label: str = ""


def expected_information_gain(epistemic_std: float, obs_sigma: float) -> float:
    """IG ≈ ½ log(1 + Var_epistemic / σ²_obs). Monotone in model disagreement."""
    var_epi = max(epistemic_std, 1e-9) ** 2
    var_obs = obs_sigma ** 2
    return 0.5 * np.log1p(var_epi / var_obs)


def select_next_observation(candidates: List[CandidateObservation],
                            epistemic: Dict[str, Dict[str, float]],
                            lam: float = 0.1) -> Tuple[CandidateObservation, float]:
    """x* = argmax IG(x) − λ·cost(x)."""
    best, best_score = None, -np.inf
    for c in candidates:
        std_e = epistemic.get(c.entity_id, {}).get(c.dim, 0.0)
        score = expected_information_gain(std_e, c.sigma) - lam * c.cost
        if score > best_score:
            best, best_score = c, score
    return best, float(best_score)


def make_candidates(world_state: WorldState,
                    dims: Tuple[str, ...] = ("ndvi", "soil_moisture", "stress"),
                    sigma: float = 0.05) -> List[CandidateObservation]:
    out = []
    for eid in world_state.history:
        for d in dims:
            if d in world_state.current(eid):
                out.append(CandidateObservation(eid, d, cost=1.0, sigma=sigma,
                                                label=f"re-observe {eid}.{d}"))
    return out
