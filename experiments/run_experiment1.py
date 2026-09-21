"""
Experiment 1 — H1: Does the world-state transition model beat baselines?

Baselines:
  1. Persistence        ŝ(t+1) = ŝ(t)
  2. Per-entity AR(1)   ŝ(t+1) = a·ŝ(t) + b   (a,b fit per entity/dim)
  3. Retrieval          ŝ(t+1) = state of most similar historical window
  4. ORBITAL ensemble   physics + coupling, no re-fit on test regions

Also runs H2 (counterfactual validity), H3 (calibration), H4 (acquisition).

Outputs: experiments/results/exp1_*.json, experiments/figures/*.png
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

from orbital.environment import GroundTruthWorld
from orbital.models import DIMS, EnsembleWorldModel
from orbital.ontology import ObjectType
from orbital.simulation import Simulator
from orbital.state import WorldState

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "experiments" / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def generate_trajectories(n_regions=8, T=80, seed=7):
    """Run ground truth, store full hidden state trajectories."""
    w = GroundTruthWorld(n_regions=n_regions, seed=seed)
    traj = {r.id: {d: [] for d in list(DIMS) + ["precipitation"]}
            for r in w.schema.by_type(ObjectType.REGION)}
    for t in range(T):
        for r in w.schema.by_type(ObjectType.REGION):
            for d in traj[r.id]:
                traj[r.id][d].append(r.state.get(d, 0.0))
        w.step()
    return w, traj


def fit_ar1(series):
    x, y = series[:-1], series[1:]
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def main():
    print("== Experiment 1: world model vs baselines ==")
    n_regions, T = 8, 80
    split = int(T * 0.7)
    w, traj = generate_trajectories(n_regions, T)

    # Fit ORBITAL ensemble on train split (assimilate weekly observations,
    # including the exogenous precipitation driver so the model can condition)
    ws = WorldState(w.schema)
    ens = EnsembleWorldModel(w.schema, n_models=8, seed=0)
    for t in range(split):
        for e in w.schema.by_type(ObjectType.REGION):
            ws.commit(e.id, {d: traj[e.id][d][t] for d in traj[e.id]})
    # calibrate per-member bias on training residuals (proper fitting)
    # season during step k of the train window: start + k/52 where
    # start = phase after T total steps, minus T/52
    train_season_start = (w.phase - T / 52) % 1.0
    ens.fit_bias(ws, traj, n_steps=split - 1, season=train_season_start)

    # ---- Evaluation over test horizon ----
    horizon = T - split
    entity = "region_0"

    # 1. persistence
    pred_persist = {d: [traj[entity][d][split - 1]] * horizon for d in DIMS}

    # 2. AR(1) per dim (fit on train)
    pred_ar = {}
    for d in DIMS:
        a, b = fit_ar1(np.array(traj[entity][d][:split]))
        out, cur = [], traj[entity][d][split - 1]
        for _ in range(horizon):
            cur = a * cur + b
            out.append(cur)
        pred_ar[d] = out

    # 3. retrieval: most similar 5-window in train, take its continuation
    win = 5
    pred_ret = {}
    test_win = np.array([traj[entity][d][split - win:split] for d in DIMS]).flatten()
    best_i, best_d = None, np.inf
    for i in range(split - win):
        cand = np.array([traj[entity][d][i:i + win] for d in DIMS]).flatten()
        dist = np.linalg.norm(cand - test_win)
        if dist < best_d:
            best_d, best_i = dist, i
    for d in DIMS:
        pred_ret[d] = traj[entity][d][best_i + win: best_i + win + horizon]

    # 4. ORBITAL: roll the ensemble from the last observed state
    sim = Simulator(w.schema, ens)
    # season of the LAST OBSERVED step (t = split-1)
    season_at_split = (train_season_start + (split - 1) / 52) % 1.0
    roll = sim.rollout(ws, horizon=horizon, season=season_at_split)
    pred_orb = {d: [s[entity][d] for s in roll.states] for d in DIMS}

    truth = {d: traj[entity][d][split: split + horizon] for d in DIMS}

    # ---- Metrics ----
    results = {"entity": entity, "horizon": horizon, "rmse": {}}
    for name, pred in [("persistence", pred_persist), ("ar1", pred_ar),
                       ("retrieval", pred_ret), ("orbital", pred_orb)]:
        results["rmse"][name] = {d: rmse(pred[d], truth[d]) for d in DIMS}
        results["rmse"][name]["mean"] = float(np.mean(
            [results["rmse"][name][d] for d in DIMS]))

    orb_mean = results["rmse"]["orbital"]["mean"]
    base_mean = results["rmse"]["persistence"]["mean"]
    results["skill_vs_persistence"] = 1 - orb_mean / base_mean

    # ---- H2: counterfactual validity ----
    from orbital.ontology import Event, EventType
    do_roll = sim.rollout(ws, horizon=horizon, interventions=[
        Event(EventType.HEATWAVE, t=0, targets=[e.id for e in w.schema.by_type(ObjectType.REGION)],
              magnitude=1.0)])
    results["counterfactual_violations"] = do_roll.violations
    results["counterfactual_steps"] = do_roll.steps

    # ---- H3: calibration (ensemble spread vs actual error) ----
    errs = np.array([[abs(roll.states[k][entity][d] - truth[d][k])
                      for d in DIMS] for k in range(horizon)])
    stds = np.array([[roll.epistemic[k][entity][d] for d in DIMS]
                     for k in range(horizon)])
    err_flat, std_flat = errs.flatten(), stds.flatten()
    # rank correlation: higher spread should accompany higher error
    results["spread_error_spearman"] = float(
        np.corrcoef(np.argsort(np.argsort(err_flat)),
                    np.argsort(np.argsort(std_flat)))[0, 1])

    # ---- Figures ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Experiment 1 — World Model vs Baselines (region_0)", fontweight="bold")
    for ax, d in zip(axes.flat, DIMS):
        ax.plot(range(split, T), truth[d], "k-", lw=2, label="ground truth")
        ax.plot(range(split - 1, T), [traj[entity][d][split - 1]] + list(pred_persist[d]),
                "--", color="#999", label="persistence")
        ax.plot(range(split, T), pred_ar[d], "--", color="#ff9100", label="AR(1)")
        ax.plot(range(split, T), pred_ret[d], ":", color="#8b949e", label="retrieval")
        ax.plot(range(split, T), pred_orb[d], "-", color="#00b8d4", lw=2, label="ORBITAL")
        ax.set_title(d)
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "exp1_worldmodel_vs_baselines.png", dpi=150)
    plt.close(fig)

    # calibration / spread-error scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(std_flat, err_flat, s=12, alpha=0.5, color="#00b8d4")
    ax.set_xlabel("ensemble epistemic std")
    ax.set_ylabel("actual |error|")
    ax.set_title(f"Uncertainty quality (rank-corr = {results['spread_error_spearman']:.2f})")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "exp1_uncertainty_quality.png", dpi=150)
    plt.close(fig)

    (RES / "exp1_results.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(results["rmse"], indent=1))
    print(f"skill vs persistence: {results['skill_vs_persistence']:+.1%}")
    print(f"counterfactual violations: {results['counterfactual_violations']}/{results['counterfactual_steps']}")
    print(f"spread<->error rank-corr: {results['spread_error_spearman']:.2f}")
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
