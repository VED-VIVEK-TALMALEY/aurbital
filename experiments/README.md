# ORBITAL Experiments

Every hypothesis is tested by a committed, re-runnable script. Results JSONs
and figures are regenerated into `results/` and `figures/`.

```bash
python experiments/run_experiment1.py   # H1 prediction skill + H2 counterfactual validity
python experiments/run_experiment2.py   # H3 calibration + H4 active perception
```

## Results (latest run)

### Experiment 1 — H1 prediction skill (24-step horizon, region_0, RMSE)

| Model | ndvi | soil_moisture | temperature | stress | mean |
|-------|------|---------------|-------------|--------|------|
| Persistence | 0.353 | 0.038 | 3.329 | 0.000 | 0.930 |
| AR(1) | 0.454 | 0.016 | 3.045 | 0.006 | 0.880 |
| Retrieval (nearest historical window) | 0.248 | 0.051 | 4.435 | 0.005 | 1.185 |
| **ORBITAL ensemble** | **0.267** | 0.033 | **1.380** | **0.000** | **0.420** |

- **Skill vs persistence: +54.9%**
- Temperature RMSE 1.38 vs 3.33 (persistence) — the shared seasonal
  climatology prior tracks the exogenous seasonal forcing.
- Spread ↔ error rank correlation (H3): **0.70**
- Constraint violations (H2): **0 / 24 steps**

![world model vs baselines](figures/exp1_worldmodel_vs_baselines.png)

### Experiment 2 — H3 calibration & H4 active perception

| Metric | Value |
|--------|-------|
| Spread ↔ error rank correlation | **0.637** |
| ECE (raw → after variance rescaling) | 0.439 → 0.220 |
| Coverage @ nominal 0.9 | 0.799 |
| Active acquisition posterior variance | 2.342 |
| Random acquisition posterior variance | 2.585 |
| **Active efficiency gain (H4)** | **+9.4%** |

![calibration](figures/exp2_calibration.png)
![active perception](figures/exp2_active_perception.png)

## Honest status

- **H1 supported**: ensemble world model beats persistence, AR(1) and
  retrieval on aggregate RMSE with a physics-informed seasonal prior.
- **H2 supported**: zero constraint violations across all rollouts.
- **H3 partially supported**: spread ranks error well (ρ≈0.64–0.70), but
  absolute calibration requires a held-out conformal calibration split
  (raw ECE is high; the rescale factor is fitted on the same points).
- **H4 directionally supported**: +9.4% variance reduction over random
  acquisition with symmetric sensor noise; heterogeneous sensors are the
  next milestone to widen the gap.
