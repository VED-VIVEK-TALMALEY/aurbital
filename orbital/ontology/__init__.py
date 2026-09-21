"""ORBITAL Ontology — typed objects, relations, events, actions.

Code-first schema of reality. See docs/ontology.md for the human contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ObjectType(str, Enum):
    REGION = "Region"
    CROP_FIELD = "CropField"
    RIVER = "River"
    LAKE = "Lake"
    FOREST = "Forest"
    ROAD = "Road"
    WEATHER_CELL = "WeatherCell"
    SETTLEMENT = "Settlement"


# Physical bounds per dynamic state dimension. Enforced by the engine.
STATE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "ndvi": (0.0, 1.0),
    "soil_moisture": (0.0, 1.0),
    "temperature": (-30.0, 55.0),
    "growth_stage": (0.0, 1.0),
    "stress": (0.0, 1.0),
    "flow_rate": (0.0, 1e5),
    "level": (0.0, 1e4),
    "canopy_density": (0.0, 1.0),
    "fire_risk": (0.0, 1.0),
    "passability": (0.0, 1.0),
    "precipitation": (0.0, 500.0),
    "humidity": (0.0, 1.0),
    "water_demand": (0.0, 1e6),
    "power_demand": (0.0, 1e6),
}


class RelationType(str, Enum):
    LOCATED_IN = "LOCATED_IN"
    FLOWS_THROUGH = "FLOWS_THROUGH"
    SUPPLIES_WATER = "SUPPLIES_WATER"
    ADJACENT_TO = "ADJACENT_TO"
    AFFECTS = "AFFECTS"
    OBSERVES = "OBSERVES"


class EventType(str, Enum):
    RAINFALL = "Rainfall"
    DROUGHT = "Drought"
    HEATWAVE = "Heatwave"
    FLOOD = "Flood"
    IRRIGATION = "Irrigation"
    HARVEST = "Harvest"


@dataclass
class Entity:
    """A typed node in the world graph."""
    id: str
    type: ObjectType
    static: Dict[str, float] = field(default_factory=dict)
    state: Dict[str, float] = field(default_factory=dict)

    def clamp_state(self) -> None:
        for k, v in self.state.items():
            if k in STATE_BOUNDS:
                lo, hi = STATE_BOUNDS[k]
                self.state[k] = max(lo, min(hi, v))


@dataclass
class Relation:
    src: str
    rel: RelationType
    dst: str


@dataclass
class Event:
    type: EventType
    t: int
    targets: List[str] = field(default_factory=list)
    magnitude: float = 1.0
    """Event application is implemented by models/transition.py."""


class WorldSchema:
    """Validates entities/relations against the ontology."""

    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

    def add_entity(self, e: Entity) -> None:
        for k in e.state:
            if k not in STATE_BOUNDS:
                raise ValueError(f"Unknown state dimension '{k}' on {e.id}")
        e.clamp_state()
        self.entities[e.id] = e

    def add_relation(self, r: Relation) -> None:
        if r.src not in self.entities or r.dst not in self.entities:
            raise ValueError(f"Relation endpoints must exist: {r.src} -> {r.dst}")
        self.relations.append(r)

    def neighbors(self, entity_id: str) -> List[Tuple[RelationType, str]]:
        out = []
        for r in self.relations:
            if r.src == entity_id:
                out.append((r.rel, r.dst))
            elif r.dst == entity_id:
                out.append((r.rel, r.src))
        return out

    def by_type(self, t: ObjectType) -> List[Entity]:
        return [e for e in self.entities.values() if e.type == t]
