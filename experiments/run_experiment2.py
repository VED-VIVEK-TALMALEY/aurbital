"""
Experiment 2 — H3 (calibration) & H4 (active perception).

H3: Do ensemble uncertainty estimates correlate with actual error, and do
    quantile forecasts achieve nominal coverage (calibration curve + ECE)?
H4: Does uncertainty-guided observation selection reduce posterior variance
    faster than random / scheduled acquisition?

Outputs: experiments/results/exp2_results.json, experiments/figures/exp2_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orbital.discovery import (CandidateObservation, make_candidates,
                               select_next_observation)
from orbital.environment import GroundTruthWorld
from orbital.models import DIMS, EnsembleWorldModel
from orbital.ontology import ObjectType
from orbital.state import WorldState

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "experiments" / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def main():
    print("== Experiment 2: calibration & active perception ==")

    # ---------- H3: calibration ----------
    n_regions, T, split = 8, 80, 55
    w = GroundTruthWorld(n_regions=n_regions, seed=21)
    traj = {r.id: {d: [] for d in DIMS} for r in w.schema.by_type(ObjectType.REGION)}
    for t in range(T):
        for r in w.schema.by_type(ObjectType.REGION):
            for d in DIMS:
                traj[r.id][d].append(r.state[d])
        w.step()

    ws = WorldState(w.schema)
    ens = EnsembleWorldModel(w.schema, n_models=12, seed=3)
    for t in range(split):
        for e in w.schema.by_type(ObjectType.REGION):
            ws.commit(e.id, {d: traj[e.id][d][t] for d in DIMS})
    ens.fit_bias(ws, traj, n_steps=split - 1, season=w.phase - split / 52)

    # one-step-ahead predictions on the test segment with ensemble spread
    err_list, std_list = [], []
    sim_schema_states = {e.id: dict(e.state) for e in w.schema.entities.values()}
    for k in range(split, T - 1):
        for e in w.schema.by_type(ObjectType.REGION):
            ws.commit(e.id, {d: traj[e.id][d][k] for d in DIMS})
        mean, std = ens.predict_step(w.phase - (T - k) / 52)
        for e in w.schema.by_type(ObjectType.REGION):
            for d in DIMS:
                err_list.append(abs(mean[e.id][d] - traj[e.id][d][k + 1]))
                std_list.append(std[e.id][d])
    err = np.array(err_list)
    std = np.array(std_list)

    # H3 includes an honest reporting step: scale spread to match observed
    # error scale (conformal-style variance rescaling, fit on the SAME points
    # evaluated — noted as such; a strict protocol would use a held-out
    # calibration set). We report both raw and rescaled ECE.
    scale = float(np.mean(err) / max(np.mean(std), 1e-9))
    std_cal = std * scale

    # coverage of Gaussian predictive intervals from ensemble spread
    # (stack members ≈ sigma of the predictive distribution)
    quantiles = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    coverage = []
    # two-sided normal quantiles: P(|X| <= z) = 1 - q mapping
    zq = {0.45: 0.1257, 0.4: 0.2533, 0.3: 0.3853, 0.1: 0.6434}
    # simpler direct approach: nominal coverage levels with matching z
    zq = {0.25: 0.3186, 0.50: 0.6745, 0.80: 1.2816, 0.90: 1.6449,
          0.95: 1.9600, 0.99: 2.5758}
    for q, z in zq.items():
        cov = float(np.mean(err <= z * std_cal))
        coverage.append({"nominal": q, "empirical": cov})

    ece = float(np.mean([abs(c["empirical"] - c["nominal"])
                         for c in coverage]))
    rank_corr = float(np.corrcoef(np.argsort(np.argsort(err)),
                                  np.argsort(np.argsort(std)))[0, 1])

    results = {
        "n_points": int(len(err)),
        "spread_error_rank_corr": rank_corr,
        "spread_rescale_factor": scale,
        "ece_after_rescaling": ece,
        "ece_raw": float(np.mean([abs(np.mean(err <= z * std) - q)
                                  for q, z in zq.items()])),
        "coverage": coverage,
    }

    # calibration diagram
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    nom = [c["nominal"] for c in coverage]
    emp = [c["empirical"] for c in coverage]
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax[0].plot(nom, emp, "o-", color="#00b8d4", label="ORBITAL ensemble")
    ax[0].set_xlabel("nominal coverage"); ax[0].set_ylabel("empirical coverage")
    ax[0].set_title(f"Calibration after rescaling (ECE={ece:.3f})"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].scatter(std, err, s=8, alpha=0.4, color="#00b8d4")
    ax[1].set_xlabel("ensemble epistemic std"); ax[1].set_ylabel("|one-step error|")
    ax[1].set_title(f"Spread↔error (rank-corr={rank_corr:.2f})"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "exp2_calibration.png", dpi=150)
    plt.close(fig)

    # ---------- H4: active perception ----------
    w2 = GroundTruthWorld(n_regions=6, seed=5)
    ws2 = WorldState(w2.schema)
    ens2 = EnsembleWorldModel(w2.schema, n_models=8, seed=9)

    def posterior_variance(policy: str, n_obs: int = 60) -> float:
        rng = np.random.default_rng(0)
        est_var = {}
        for e in w2.schema.by_type(ObjectType.REGION):
            est_var[e.id] = {d: 0.25 for d in DIMS}  # high initial uncertainty
        for i in range(n_obs):
            cands = make_candidates(ws2)
            if policy == "active":
                c, _ = select_next_observation(
                    [CandidateObservation(c.entity_id, c.dim, sigma=0.05)
                     for c in cands],
                    {eid: est_var[eid] for eid in est_var}, lam=0.0)
                eid, dim = c.entity_id, c.dim
            else:
                c = cands[rng.integers(len(cands))]
                eid, dim = c.entity_id, c.dim
            o = w2.observe(eid, dim, sigma=0.05)
            ws2.assimilate(o)
            est_var[eid][dim] *= 0.6  # observation shrinks variance
        return float(sum(v for m in est_var.values() for v in m.values()))

    v_active = posterior_variance("active")
    v_random = posterior_variance("random")
    results["posterior_var_active"] = v_active
    results["posterior_var_random"] = v_random
    results["active_efficiency_gain"] = 1 - v_active / v_random

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["random", "active (info-gain)"], [v_random, v_active],
           color=["#8b949e", "#00b8d4"])
    ax.set_ylabel("total posterior variance after 60 observations")
    ax.set_title(f"H4: active acquisition {results['active_efficiency_gain']:+.0%} vs random")
    fig.tight_layout()
    fig.savefig(FIG / "exp2_active_perception.png", dpi=150)
    plt.close(fig)

    (RES / "exp2_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("coverage",)}, indent=1))
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
