"""ORBITAL Reasoning — contradiction detection and grounded claim checking.

The LLM interface is intentionally NOT here at the center. This module lets
claims survive *evidence checks* before anything is verbalized. The LLM (when
used) only translates the structured verdict into natural language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

CHI2_99 = 6.63  # 1 dof


@dataclass
class Evidence:
    source: str
    entity_id: str
    dim: str
    value: float
    sigma: float


@dataclass
class Verdict:
    status: str  # "CONSISTENT" | "CONFLICT"
    claims: List[str] = field(default_factory=list)
    conflicts: List[Tuple[str, str, float]] = field(default_factory=list)
    recommendation: Optional[str] = None


def disagreement(e1: Evidence, e2: Evidence) -> float:
    """D = (μ1−μ2)² / (σ1²+σ2²). > χ²(0.99) ⇒ conflict."""
    return (e1.value - e2.value) ** 2 / (e1.sigma ** 2 + e2.sigma ** 2)


def detect_contradictions(evidence: List[Evidence]) -> Verdict:
    """Pairwise standardized disagreement among same (entity, dim) evidence."""
    v = Verdict(status="CONSISTENT")
    grouped: Dict[Tuple[str, str], List[Evidence]] = {}
    for e in evidence:
        grouped.setdefault((e.entity_id, e.dim), []).append(e)

    for (eid, dim), evs in grouped.items():
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                d = disagreement(evs[i], evs[j])
                if d > CHI2_99:
                    v.status = "CONFLICT"
                    v.conflicts.append((f"{evs[i].source}:{evs[i].value:.2f}",
                                        f"{evs[j].source}:{evs[j].value:.2f}", d))
        for e in evs:
            v.claims.append(f"{eid}.{dim} = {e.value:.2f} [{e.source} ±{e.sigma}]")

    if v.status == "CONFLICT":
        v.recommendation = ("Withhold fused answer; acquire disambiguating "
                            "observation with highest information gain.")
    return v


def check_claim_grounding(claim_value: float, evidence: List[Evidence],
                          tolerance: float = 2.0) -> str:
    """Is a generated claim (numeric) supported by sensor evidence?

    Returns SUPPORTED / UNSUPPORTED. This is the Reality-Consistency check:
    generation must agree with measurements, not the other way around.
    """
    if not evidence:
        return "UNSUPPORTED (no evidence)"
    mu = np.mean([e.value for e in evidence])
    sig = np.sqrt(sum(e.sigma ** 2 for e in evidence)) + 1e-9
    z = abs(claim_value - mu) / sig
    return "SUPPORTED" if z <= tolerance else f"UNSUPPORTED (z={z:.1f})"
