"""ORBITAL Environment — a synthetic-but-principled ground-truth world.

Used to generate (state, observation) data with known dynamics so hypotheses
H1–H4 are testable: we can score predictions against *true* hidden states,
which is impossible with real data alone.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from orbital.ontology import (Entity, EventType, ObjectType, Relation,
                              RelationType, WorldSchema)
from orbital.models import DIMS, apply_events, physics_delta, seasonal_temp
from orbital.state import Observation


class GroundTruthWorld:
    """The real world (hidden). Observers only see noisy samples."""

    def __init__(self, n_regions: int = 10, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.season = self.rng.uniform(0, 1)
        self.schema = self._build(n_regions)

    @property
    def phase(self) -> float:
        """Current seasonal phase (for aligning model rollouts)."""
        return self.season

    def _build(self, n: int) -> WorldSchema:
        schema = WorldSchema()
        # weather cells + regions + rivers
        for i in range(2):
            schema.add_entity(Entity(
                f"weather_{i}", ObjectType.WEATHER_CELL,
                static={"grid_id": i},
                state={"precipitation": self.rng.uniform(0, 5),
                       "temperature": self.rng.uniform(15, 30),
                       "humidity": self.rng.uniform(.3, .9)}))
        for i in range(n):
            lat, lon = 20 + self.rng.uniform(0, 8), 70 + self.rng.uniform(0, 10)
            schema.add_entity(Entity(
                f"region_{i}", ObjectType.REGION,
                static={"lat": lat, "lon": lon, "area_km2": self.rng.uniform(50, 500)},
                state={"ndvi": self.rng.uniform(.2, .8),
                       "soil_moisture": self.rng.uniform(.2, .8),
                       "temperature": self.rng.uniform(15, 32),
                       "stress": self.rng.uniform(0, .3)}))
        for i in range(2):
            schema.add_entity(Entity(
                f"river_{i}", ObjectType.RIVER,
                static={"width_m": self.rng.uniform(5, 60), "order": 3},
                state={"flow_rate": self.rng.uniform(10, 80),
                       "level": self.rng.uniform(1, 5)}))
        # relations: each region fed by a river, adjacent chain
        for i in range(n):
            schema.add_relation(Relation(f"river_{i % 2}", RelationType.FLOWS_THROUGH, f"region_{i}"))
            schema.add_relation(Relation(f"region_{i}", RelationType.LOCATED_IN, f"weather_{i % 2}"))
            if i > 0:
                schema.add_relation(Relation(f"region_{i-1}", RelationType.ADJACENT_TO, f"region_{i}"))
        return schema

    def step(self) -> None:
        """Advance the TRUE world one week."""
        self.t += 1
        self.season = (self.season + 1/52) % 1.0
        # exogenous weather
        for w in self.schema.by_type(ObjectType.WEATHER_CELL):
            w.state["temperature"] = seasonal_temp(self.season) + self.rng.normal(0, 2)
            rain_p = .5 + .3 * np.sin(2 * np.pi * self.season + 1)
            w.state["precipitation"] = float(self.rng.random() < rain_p) * self.rng.gamma(2, 4)
        # regions respond — same semi-mechanistic prior form as the model
        # (correctly specified prior; H1 tests estimation under noise, not
        # a deliberate prior mismatch)
        regions = self.schema.by_type(ObjectType.REGION)
        for i, r in enumerate(regions):
            w = self.schema.entities[f"weather_{i % 2}"]
            rain = w.state["precipitation"]
            temp = w.state["temperature"]
            # exogenous drivers are part of the (observable) region state so
            # the model can condition on them like any other observation
            r.state["precipitation"] = rain
            d = physics_delta(r.state, self.season)
            r.state["soil_moisture"] = float(np.clip(
                r.state["soil_moisture"] + d["soil_moisture"] +
                0.01 * self.rng.normal(), 0, 1))
            d = physics_delta(r.state, self.season)  # recompute post-sm update
            r.state["stress"] = float(np.clip(
                r.state["stress"] + d["stress"] + 0.01 * self.rng.normal(), 0, 1))
            d = physics_delta(r.state, self.season)
            r.state["ndvi"] = float(np.clip(
                r.state["ndvi"] + d["ndvi"] + 0.005 * self.rng.normal(), 0, 1))
            r.state["temperature"] = temp + self.rng.normal(0, .5)
        # rivers respond to upstream weather
        for i, rv in enumerate(self.schema.by_type(ObjectType.RIVER)):
            up = self.schema.entities[f"weather_{i}"]
            rv.state["flow_rate"] = float(np.clip(
                0.8 * rv.state["flow_rate"] + 5 * up.state["precipitation"] + self.rng.normal(0, 2), 0, 1e5))
            rv.state["level"] = float(np.clip(0.9 * rv.state["level"] + 0.02 * rv.state["flow_rate"], 0, 1e4))

    def observe(self, entity_id: str, dim: str, sigma: float = 0.05) -> Observation:
        true_val = self.schema.entities[entity_id].state[dim]
        return Observation(entity_id, dim,
                           float(true_val + self.rng.normal(0, sigma)),
                           self.t, source="sensor", sigma=sigma)
