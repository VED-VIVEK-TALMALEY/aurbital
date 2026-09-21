# ORBITAL — World-State Intelligence & Simulation Engine

Research question:

> Can heterogeneous observations of the physical world be converted into a
> persistent, causal, executable computational model that supports prediction,
> counterfactual simulation, calibrated uncertainty, and optimal selection of
> the next observation — measurably outperforming retrieval-based and purely
> generative alternatives?

The LLM is deliberately **not** at the center: it is a replaceable interface at
the reasoning edge. The research artifact is the world model itself.

## Architecture

![ORBITAL architecture](../experiments/figures/orbital_architecture.png)

## Engine Layout

| Module | Purpose |
|--------|---------|
| `ontology/` | Typed objects, relations, events, physical bounds (schema of reality) |
| `state/` | Persistent world state + Kalman-style observation assimilation |
| `models/` | Physics-informed transition dynamics + deep-ensemble epistemic uncertainty |
| `simulation/` | Forward rollouts, do-operator counterfactuals, constraint checking |
| `discovery/` | Expected-information-gain active perception |
| `reasoning/` | Contradiction detection, claim grounding before verbalization |
| `environment/` | Ground-truth world generator (falsifiable testbed with known hidden states) |
| `docs/` | Research problem, mathematical formulation, ontology specification |

## Hypotheses and Results

All results come from committed scripts on the synthetic ground-truth
environment (hidden true states are known, so error is measurable). Re-run:

```bash
python experiments/run_experiment1.py
python experiments/run_experiment2.py
```

### H1 — Prediction Skill

24-step horizon, region_0, RMSE:

| Model | ndvi | soil_moisture | temperature | stress | mean |
|-------|------|---------------|-------------|--------|------|
| Persistence | 0.353 | 0.038 | 3.329 | 0.000 | 0.930 |
| AR(1) | 0.454 | 0.016 | 3.045 | 0.006 | 0.880 |
| Retrieval (nearest historical window) | 0.248 | 0.051 | 4.435 | 0.005 | 1.185 |
| **ORBITAL ensemble** | **0.267** | 0.033 | **1.380** | **0.000** | **0.420** |

**Skill vs persistence: +54.9%.** The shared seasonal climatology prior cuts
temperature error to less than half of every baseline.

![World model vs baselines](../experiments/figures/exp1_worldmodel_vs_baselines.png)

### H2 — Counterfactual Validity

| Check | Result |
|-------|--------|
| Physical-constraint violations under do(HEATWAVE), 24 steps, all regions | **0** |
| Counterfactual divergence (soil moisture drops under heatwave) | Yes, within bounds |

### H3 — Uncertainty Quality

| Metric | Value |
|--------|-------|
| Ensemble spread vs actual error, rank correlation | **0.70** |
| Coverage at nominal 0.9 | 0.799 |
| ECE raw | 0.439 |
| ECE after conformal-style variance rescaling | **0.220** |

Honestly reported limitation: the rescale factor is fitted on the same points;
a strict protocol would use a held-out calibration split. **H3 is partially
supported** — spread ranks error well, absolute calibration needs the
calibration split.

![Calibration](../experiments/figures/exp2_calibration.png)

### H4 — Active Perception

| Policy | Posterior variance (60 observations) |
|--------|--------------------------------------|
| Random acquisition | 2.585 |
| Uncertainty-guided (info-gain) | **2.342** |
| **Efficiency gain** | **+9.4%** |

Directionally correct; below the 30% target because current candidates share
symmetric sensor noise. The mechanism
(IG = 0.5 * log(1 + Var_epi / sigma_obs^2)) is in place; heterogeneous sensor
costs/noise are the next milestone.

![Active perception](../experiments/figures/exp2_active_perception.png)

## Quick Start

```bash
pip install numpy matplotlib
python -m unittest tests.test_orbital          # 11 engine tests
python experiments/run_experiment1.py          # H1 + H2 + figures
python experiments/run_experiment2.py          # H3 + H4 + figures
python orbital/generate_architecture_diagram.py
```

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| Falsifiable | Every claim maps to a committed script + metric; no diagram stands in for evidence |
| LLM at the edge | `reasoning/` checks claims against evidence before verbalization; unsupported claims are flagged |
| Physics-informed, not physics-only | Semi-mechanistic priors (seasonal vegetation capacity, moisture-stress coupling) + ensemble residuals |
| Constraints everywhere | Every rollout step is bounds-checked (0 violations across all experiments) |

## Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| 1 | World-state representation + ontology | Done |
| 2 | Transition model + ensemble uncertainty | Done |
| 3 | Counterfactual simulator with constraint validity | Done |
| 4 | Contradiction detection + claim grounding | Done |
| 5 | Active perception machinery + first experiments | Done |
| 6 | Heterogeneous sensors (SAR, cloud-gapped) for a wider H4 gap | Planned |
| 7 | Couple to TerraSight SpectralViT: satellite-derived NDVI assimilated as observations | Planned |
| 8 | Geographic generalization (train region A, test region B) | Planned |
| 9 | Decision layer (interventions under cost constraints) | Planned |
| 10 | Public benchmark (region / sensor / temporal shift splits) | Planned |

## Relationship to TerraSight

TerraSight (the `earthaware/` layer) answers "what is in this image?".
ORBITAL answers "what will happen, what if, how sure are we, and what should
we measure next?". Phase 7 connects them: the trained SpectralViT becomes a
sensor whose outputs are assimilated as observations.
