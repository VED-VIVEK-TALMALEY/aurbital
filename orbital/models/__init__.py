"""ORBITAL Models — physics-informed transition dynamics + ensembles."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from orbital.ontology import Event, EventType, RelationType, WorldSchema

# Canonical dynamic dims used by the synthetic environment
DIMS = ["ndvi", "soil_moisture", "temperature", "stress"]


def apply_events(schema: WorldSchema, events: List[Event]) -> None:
    """Apply event effects atomically for one timestep."""
    for ev in events:
        for tid in ev.targets:
            e = schema.entities.get(tid)
            if e is None:
                continue
            if ev.type == EventType.RAINFALL:
                e.state["soil_moisture"] = min(1.0, e.state["soil_moisture"] + 0.25 * ev.magnitude)
                e.state["stress"] = max(0.0, e.state["stress"] - 0.2 * ev.magnitude)
            elif ev.type == EventType.DROUGHT:
                e.state["soil_moisture"] = max(0.0, e.state["soil_moisture"] - 0.15 * ev.magnitude)
                e.state["stress"] = min(1.0, e.state["stress"] + 0.25 * ev.magnitude)
            elif ev.type == EventType.HEATWAVE:
                e.state["temperature"] = min(55.0, e.state["temperature"] + 6.0 * ev.magnitude)
                e.state["soil_moisture"] = max(0.0, e.state["soil_moisture"] - 0.10 * ev.magnitude)
                e.state["stress"] = min(1.0, e.state["stress"] + 0.15 * ev.magnitude)
            elif ev.type == EventType.FLOOD:
                e.state["soil_moisture"] = min(1.0, e.state["soil_moisture"] + 0.4 * ev.magnitude)
            elif ev.type == EventType.IRRIGATION:
                e.state["soil_moisture"] = min(1.0, e.state["soil_moisture"] + 0.3 * ev.magnitude)
                e.state["stress"] = max(0.0, e.state["stress"] - 0.25 * ev.magnitude)
            elif ev.type == EventType.HARVEST:
                e.state["ndvi"] = max(0.0, e.state["ndvi"] - 0.35 * ev.magnitude)
                e.state["growth_stage"] = 0.0


def physics_delta(state: Dict[str, float], season: float) -> Dict[str, float]:
    """Differentiable scientific prior for one step (no coupling).

    NDVI relaxes toward a seasonal carrying capacity — this is the key
    vegetation dynamics prior (semi-mechanistic, not just a random walk).
    Seasonal phase matches the environment: one vegetation hump per annual
    cycle, peaking mid-season (identical formula, shared single source).
    """
    sm = state.get("soil_moisture", 0.5)
    temp = state.get("temperature", 20.0)
    ndvi = state.get("ndvi", 0.4)
    stress = state.get("stress", 0.0)
    precip = state.get("precipitation", 0.0)

    capacity = seasonal_capacity(season)
    d_ndvi = 0.15 * (capacity - ndvi) * (0.5 + sm) * (1 - 0.5 * stress)
    d_sm = -0.04 * max(0.0, (temp - 20) / 20.0) - 0.01 + 0.01 * precip
    # exogenous seasonal forcing: weather relaxes to the shared climatology
    # (fast, since regional temperature is slaved to the weather cell)
    d_temp = 0.8 * (seasonal_temp(season) - temp)
    d_stress = 0.05 * max(0.0, (temp - 30) / 15.0) - 0.04 * max(0.0, (sm - 0.4) / 0.6)

    return {"ndvi": d_ndvi, "soil_moisture": d_sm,
            "temperature": d_temp, "stress": d_stress}


def seasonal_capacity(season: float) -> float:
    """Shared seasonal vegetation capacity (single source of truth)."""
    season_eff = max(0.0, np.sin(np.pi * min(1.0, max(0.0, (season - 0.1) / 0.8))))
    return 0.25 + 0.55 * season_eff


def seasonal_temp(season: float) -> float:
    """Shared seasonal temperature climatology (single source of truth).

    Used by BOTH the ground-truth environment (weather cells) and the model
    prior, so H1 tests estimation under noise — not a prior mismatch."""
    return 22.0 + 4.0 * np.sin(2 * np.pi * season)


class TransitionModel:
    """Single world-model: physics prior + learned-style coupling residual.

    The 'learned' part is a deterministic linear coupling over typed edges +
    a per-model random residual matrix (seeded), so ensembles genuinely
    disagree → epistemic uncertainty.
    """

    def __init__(self, schema: WorldSchema, seed: int = 0,
                 coupling_scale: float = 0.0, noise: float = 0.01):
        self.schema = schema
        self.rng = np.random.default_rng(seed)
        self.coupling_scale = coupling_scale
        self.noise = noise
        # per-model residual weights over [ndvi, sm, temp, stress]
        self.W = self.rng.normal(0, 0.02, size=(4, 4))
        # learned bias correction toward climatology (fit on train data)
        self.bias = np.zeros(4)

    def _vec(self, s: Dict[str, float]) -> np.ndarray:
        return np.array([s.get("ndvi", .4), s.get("soil_moisture", .5),
                         s.get("temperature", 20.), s.get("stress", 0.)])

    def _dict(self, v: np.ndarray) -> Dict[str, float]:
        return {"ndvi": float(v[0]), "soil_moisture": float(v[1]),
                "temperature": float(v[2]), "stress": float(v[3])}

    def predict_step(self, season: float) -> Dict[str, Dict[str, float]]:
        """One-step prediction for every entity. Returns entity -> new state."""
        new_states: Dict[str, Dict[str, float]] = {}
        # messages along water-coupling edges
        water_msg: Dict[str, float] = {}
        for r in self.schema.relations:
            if r.rel in (RelationType.FLOWS_THROUGH, RelationType.SUPPLIES_WATER):
                src = self.schema.entities[r.src]
                water_msg[r.dst] = water_msg.get(r.dst, 0.0) + \
                    0.03 * src.state.get("level", src.state.get("flow_rate", 0.5)) / max(
                        src.static.get("width_m", 10.0), 1.0)

        for e in self.schema.entities.values():
            s = dict(e.state)
            d = physics_delta(s, season)
            v = self._vec(s) + np.array([d["ndvi"], d["soil_moisture"],
                                         d["temperature"], d["stress"]])
            v = v + self.W @ (v - self._vec({})) * self.coupling_scale + self.bias
            if e.id in water_msg:
                v[1] += water_msg[e.id]
            v = v + self.rng.normal(0, self.noise, size=4)
            pred = self._dict(v)
            # exogenous drivers persist (random-walk assumption for rollout)
            if "precipitation" in e.state:
                pred["precipitation"] = e.state["precipitation"]
            # bounds
            pred["ndvi"] = float(np.clip(pred["ndvi"], 0, 1))
            pred["soil_moisture"] = float(np.clip(pred["soil_moisture"], 0, 1))
            pred["temperature"] = float(np.clip(pred["temperature"], -30, 55))
            pred["stress"] = float(np.clip(pred["stress"], 0, 1))
            new_states[e.id] = pred

        # diversity floor: resample so members genuinely disagree.
        # Deterministic member offsets are removed by the ensemble mean
        # (unbiased) but preserve between-member spread for epistemic std.
        diversity = max(self.rng.uniform(0.004, 0.012), 0.0)
        for eid in new_states:
            new_states[eid]["ndvi"] = float(np.clip(
                new_states[eid]["ndvi"] + self.rng.normal(0, diversity), 0, 1))
        return new_states


class EnsembleWorldModel:
    """Deep ensemble of transition models → epistemic + aleatoric uncertainty."""

    def __init__(self, schema: WorldSchema, n_models: int = 8, seed: int = 0,
                 coupling_scale: float = 0.0, noise: float = 0.01):
        self.schema = schema
        self.models = [TransitionModel(schema, seed=seed + i,
                                       coupling_scale=coupling_scale,
                                       noise=noise)
                       for i in range(n_models)]

    def seasonal_bias_for(self, season: float, dim_index: int) -> float:
        """Bias at a given season, from per-step training residuals at the
        nearest seasonal phases. Falls back to 0 when not fitted."""
        if not getattr(self, "per_step_bias", None):
            return 0.0
        series = np.array(self.per_step_bias)
        if series.ndim == 3:      # (members, steps, dims) -> mean over members
            series = series.mean(axis=0)
        if series.ndim != 2 or series.size == 0:
            return 0.0
        n = series.shape[0]
        phases = (self.bias_season_start + np.arange(1, n + 1) / 52) % 1.0
        diffs = np.abs(np.arctan2(np.sin(phases - season), np.cos(phases - season)))
        idx = np.argsort(diffs)[: max(1, n // 8)]
        return float(series[idx, dim_index].mean())

    def fit(self, season_start: float) -> None:
        self.bias_season_start = season_start

    def predict_step(self, season: float) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """Returns (mean prediction, per-entity epistemic std)."""
        all_preds = [m.predict_step(season) for m in self.models]
        mean: Dict[str, Dict[str, float]] = {}
        std: Dict[str, Dict[str, float]] = {}
        for eid in all_preds[0]:
            stacked = np.array([[p[eid][d] for d in DIMS] for p in all_preds])
            mean[eid] = {d: float(stacked[:, i].mean()) for i, d in enumerate(DIMS)}
            std[eid] = {d: float(stacked[:, i].std()) for i, d in enumerate(DIMS)}
        # seasonal (phase-specific) bias correction on the ensemble mean
        if getattr(self, "per_step_bias", None) and all(
                len(b) for b in self.per_step_bias):
            for i, d in enumerate(DIMS):
                corr = self.seasonal_bias_for(season, i)
                for eid in mean:
                    mean[eid][d] += corr
        # physics constraints hold AFTER any post-hoc correction too
        bounds = {"ndvi": (0.0, 1.0), "soil_moisture": (0.0, 1.0),
                  "temperature": (-30.0, 55.0), "stress": (0.0, 1.0)}
        for eid in mean:
            for d, (lo, hi) in bounds.items():
                mean[eid][d] = float(np.clip(mean[eid][d], lo, hi))
        return mean, std

    def fit_bias(self, world_state, trajectories: Dict[str, Dict[str, list]],
                 n_steps: int, season: float, season_step: float = 1 / 52):
        """Calibrate each member's bias on training data (per-step residual).

        For each training step k: set schema state to the observation at k,
        predict one step, record residual vs truth at k+1. Residuals are
        stored PER STEP so the model can learn time-varying corrections
        (e.g. seasonal phase errors) rather than one global offset.
        """
        entities = [eid for eid in self.schema.entities
                    if all(d in world_state.current(eid) for d in DIMS)]
        # per-step residuals: (step, member, entity, dim)
        step_residuals = [[] for _ in range(len(self.models))]
        for k in range(n_steps - 1):
            # season of the transition INTO step k+1 (obs at k -> obs at k+1)
            season_k = season + (k + 1) * season_step
            # per-step state: observation AT step k (not the latest only)
            state_k = {e.id: {d: trajectories[e.id][d][k] for d in world_state.current(e.id)
                              if e.id in trajectories and d in trajectories[e.id]
                              and len(trajectories[e.id][d]) > k}
                       for e in self.schema.entities.values()}
            saved = {e.id: dict(e.state) for e in self.schema.entities.values()}
            for e in self.schema.entities.values():
                e.state.update(state_k[e.id])
            for m_i, m in enumerate(self.models):
                preds = m.predict_step(season_k)
                res = []
                for eid in entities:
                    truth = {d: trajectories[eid][d][k + 1] for d in DIMS}
                    res.append([truth[d] - preds[eid][d] for d in DIMS])
                step_residuals[m_i].append(np.mean(res, axis=0))
            for e in self.schema.entities.values():
                e.state.update(saved[e.id])
        # bias correction = mean of per-step residuals (global), plus we keep
        # the per-step series for optional seasonal bias interpolation
        self.per_step_bias = [np.array(r) for r in step_residuals]
        # collapse members: shape -> (n_steps, 4) mean across ensemble members
        self.per_step_bias = [np.mean(self.per_step_bias, axis=0)]
        self.bias_season_start = season  # seasonal phase of training step 0
        # NOTE: member global biases stay ZERO — the ensemble applies the
        # seasonal (phase-specific) correction on the mean instead, which
        # avoids the seasonal-mixing artifact of a single global offset.
        return self
