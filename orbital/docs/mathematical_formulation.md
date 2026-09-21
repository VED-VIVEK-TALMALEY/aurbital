# ORBITAL — Mathematical Formulation

## 1. Entities and State

An entity `i` at time `t` has state vector:

```
s_i(t) = [x_i, z_i(t)]
```

- `x_i` — static attributes (location, area, type)
- `z_i(t) ∈ ℝ^d` — dynamic state (ndvi, soil_moisture, temperature, ...)

World state `S(t) = (G, {s_i(t)})` where `G = (V, E)` is the typed ontology
graph (entities as nodes, typed relations as edges).

## 2. Transition Model

We separate physics-informed structure from learned residuals:

```
z_i(t+1) = z_i(t) + Δz_phys(z_i, E_t) + g_θ(z_i, NB(i), E_t) + ε,   ε ~ N(0, Σ)
```

- `Δz_phys` — differentiable scientific prior, e.g. for vegetation:
  `d(ndvi)/dt = α·(soil_moisture) − β·(temperature − T_opt)⁺ − γ·stress`
- `g_θ` — learned residual (small MLP / GNN over entity + neighbor messages)
- `NB(i)` — neighbor entities in the ontology graph (river feeds crops, etc.)
- Coupling flows along typed edges: message `m_{j→i} = φ_θ(z_j, rel_type(j,i))`

## 3. Counterfactuals & Interventions

Interventions use the **do-operator** on exogenous variables:

```
S(t+1..t+k) | do(E_t = e')   —   replace E in the rollout, keep f fixed
```

Validity constraints (physically grounded, checked every rollout step):

```
0 ≤ ndvi ≤ 1,   0 ≤ soil_moisture ≤ 1,   temp ∈ [-30, 55],   flow ≥ 0
```

Counterfactual effect at horizon k:

```
CFE_k = E[ z(t+k) | do(e') ] − z_obs(t+k)
```

## 4. Uncertainty

Deep-ensemble world model: `M` transition models `f_θ1..θM` trained with
different seeds/data orders. Predictive distribution:

```
p(z_i(t+1) | O_1..t) ≈ (1/M) Σ_m N( f_θm(z_i), Σ )
```

Decomposition:

- **Epistemic** (model disagreement): `Var_epistemic = Var_m[ f_θm(z_i) ]`
- **Aleatoric** (process noise): `E_m[ Σ_m ]` (learned per-state noise)

Calibration is measured with the **Expected Calibration Error** over quantile
forecasts: for nominal coverage `q`, empirical coverage should equal `q`.

## 5. Information Gain & Active Perception

Given candidate observations `x ∈ X` (e.g. "re-observe region 3", "add SAR
pass on region 7", "wait"), each reduces state uncertainty differently.

Expected information gain for candidate x:

```
IG(x) = H[z] − E_{o~p(o|x)}[ H[z | o, x] ]
      ≈ (1/2) · log( 1 + Var_epistemic(z) / σ²_obs(x) )
```

The active policy selects:

```
x* = argmax_{x∈X} IG(x) − λ·cost(x)
```

## 6. Contradiction Detection

Each evidence source e produces a belief `b_e = (μ_e, Σ_e)`. For sources
touching the same state dimension, compute standardized disagreement:

```
D(e1,e2) = (μ_e1 − μ_e2)² / (Σ_e1 + Σ_e2)
```

`D > χ²_1(0.99) ≈ 6.63` ⇒ **CONFLICT** — the system withholds a fused answer,
reports the conflicting evidence, and requests the highest-IG disambiguating
observation.

## 7. Evaluation Metrics

| Quantity | Definition |
|----------|-----------|
| Prediction RMSE | `sqrt(mean||Ŝ−S||²)` on held-out rollouts |
| Skill vs baseline | `1 − RMSE_model / RMSE_baseline` |
| Counterfactual validity | fraction of rollout steps violating constraints |
| ECE | `Σ_q w_q · |coverage_q − q|` over quantiles q |
| Acquisition efficiency | uncertainty (posterior variance) after N observations, active vs random |
