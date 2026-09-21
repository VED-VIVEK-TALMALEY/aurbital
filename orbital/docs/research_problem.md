# ORBITAL — Research Problem & Program

## 1. The Research Question

> **Can heterogeneous observations of the physical world be converted into a
> persistent, causal, and executable computational model — a *world state* —
> that supports prediction, counterfactual simulation, uncertainty
> quantification, and optimal selection of the next observation, measurably
> outperforming retrieval-based and purely generative alternatives?**

This is deliberately *not* "can an LLM answer questions about satellite
images." The LLM is an **interface**, not the brain. The research artifact is
the world model itself.

## 2. Why this is different from TerraSight (perception layer)

| TerraSight | ORBITAL |
|------------|---------|
| image → prediction | region → **persistent state** evolving over time |
| answers questions | answers **"what will happen"** and **"what if"** |
| correlation via VQA | **causal structure + interventions** |
| single prediction | prediction + **calibrated uncertainty** |
| passive observation | **active** selection of next best observation |
| LLM at the center | LLM at the **edge** (translator only) |

## 3. Formal Problem Statement

Let a region of the world be modeled at discrete times `t ∈ {0,1,2,...}`.

**World state:**

```
S_t = { s_1(t), s_2(t), ..., s_n(t) }
```

where each entity `s_i` carries a typed state vector (e.g. a CropField has
NDVI, soil_moisture, growth_stage, stress).

**Transition model (the core learned/scientific object):**

```
S_{t+1} = f(S_t, E_t, A_t) + ε_t
```

- `E_t` — exogenous forcing (weather, season)
- `A_t` — interventions / actions (irrigation, deforestation)
- `ε_t` — process noise

**Observation model:**

```
O_t = h(S_t) + η_t
```

Observations are noisy, partial views of the true hidden state.

**Four capabilities to be established experimentally:**

1. **Prediction** — minimize `|| Ŝ_{t+1} − S_{t+1} ||` on held-out futures.
2. **Counterfactuals** — estimate `P(S_{t+k} | do(E_t = e'))` vs observed `e`.
3. **Uncertainty** — calibrated confidence: predicted intervals should match
   empirical error rates (ECE measured).
4. **Active perception** — choose the next observation
   `x* = argmax_x 𝔼[InformationGain(x)]` and reduce uncertainty faster than
   random or scheduled acquisition.

## 4. Hypotheses (falsifiable)

- **H1:** A persistent world-state transition model predicts future regional
  state better than (a) last-value persistence, (b) per-entity AR baselines,
  and (c) retrieval of most-similar historical sequences.
- **H2:** Counterfactual rollouts under synthetic interventions remain
  physically consistent (mass/energy bounds) while diverging realistically
  from the observed trajectory.
- **H3:** Uncertainty estimates from ensemble world models are calibrated
  (ECE < 0.1 on held-out regions) and correlate with actual error.
- **H4:** Uncertainty-guided observation selection reduces expected
  posterior entropy faster than random or uniform acquisition schedules.

## 5. Success Criteria

| Capability | Metric | Target |
|------------|--------|--------|
| Prediction | RMSE vs persistence baseline | ≥ 25% improvement |
| Counterfactual | constraint violation rate | 0 violations |
| Uncertainty | Expected Calibration Error | < 0.10 |
| Active perception | uncertainty reduction vs random | ≥ 30% faster |

## 6. Non-Goals

- Not building a chatbot, dashboard, or map UI in this layer.
- Not fine-tuning LLMs. The LLM interface (`reasoning/llm_interface.py`)
  is a thin translation layer, replaceable, and not a research contribution.
- Not claiming planetary scale: a 10-region synthetic+semi-real environment
  with real satellite-derived time series is enough for falsifiable results.

## 7. Relationship to prior art

- **Palantir Ontology/Scenario** — architectural inspiration (objects,
  relations, actions, what-if worlds). ORBITAL adds: learned transition
  dynamics, calibrated uncertainty, and information-gain-driven acquisition,
  evaluated with controlled experiments.
- **NASA Prithvi / Earth-FM** — perception foundation models; ORBITAL sits
  one layer above them (state estimation, dynamics, decision).
- **Digital twins** — we adopt the state-transition formalism but focus on
  *learned* dynamics + *falsifiable experiments*, not 3D visualization.
