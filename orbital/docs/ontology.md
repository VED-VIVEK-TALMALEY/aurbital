# ORBITAL — Earth Ontology Specification

The ontology is the *schema of reality*: typed objects, typed relations,
events, and actions. It is code-first (`orbital/ontology/`); this document is
the human-readable contract.

## 1. Object Types

| Type | Static attrs | Dynamic state (d-dim) |
|------|-------------|----------------------|
| `Region` | id, lat, lon, area_km2 | ndvi, soil_moisture, temperature |
| `CropField` | crop_type, area_ha | ndvi, soil_moisture, growth_stage, stress |
| `River` | width_m, order | flow_rate, level, turbidity |
| `Lake` | volume_m3 | level, surface_temp |
| `Forest` | dominant_species | canopy_density, ndvi, fire_risk |
| `Road` | class, lanes | passability |
| `WeatherCell` | grid_id | precipitation, temperature, humidity |
| `Settlement` | population | water_demand, power_demand |

All dynamic state is bounded (see physics constraints in the formulation doc).

## 2. Relation Types (typed edges)

| Relation | Domain → Range | Coupling semantics |
|----------|---------------|-------------------|
| `LOCATED_IN` | any → Region | inherits regional exogenous forcing |
| `FLOWS_THROUGH` | River → Region | raises soil_moisture of downstream fields |
| `SUPPLIES_WATER` | Lake/River → CropField/Settlement | irrigation input |
| `ADJACENT_TO` | Region ↔ Region | spatial diffusion of temperature/stress |
| `AFFECTS` | Event → any | event-driven state jumps |
| `OBSERVES` | Sensor/Satellite → any | provenance of observations |

## 3. Events

| Event | Effect signature |
|-------|-----------------|
| `Rainfall` | +soil_moisture (all affected fields), +river flow |
| `Drought` | −soil_moisture trend, +stress |
| `Heatwave` | +temperature, −soil_moisture, +stress |
| `Flood` | +soil_moisture (saturate), −passability(roads) |
| `Irrigation` (action) | +soil_moisture targeted field |
| `Harvest` (action) | ndvi → baseline, growth_stage reset |

## 4. Actions (decision layer vocabulary)

`Observe(entity, sensor)`, `Irrigate(field, amount)`, `Evacuate(zone)`,
`Inspect(entity)`, `Simulate(scenario, horizon)`.

## 5. Invariants (enforced by the engine)

1. Every dynamic state stays within its physical bounds.
2. Causal links only along typed relations (no free-form edges).
3. Every observation records provenance (`source, time, σ_noise`).
4. Events are applied atomically per timestep.
