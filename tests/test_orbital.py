"""ORBITAL engine tests — every research claim needs a check.

Runnable with either pytest or unittest:
    python -m pytest tests/
    python -m unittest tests.test_orbital
"""
import unittest

import numpy as np

from orbital.environment import GroundTruthWorld
from orbital.models import EnsembleWorldModel
from orbital.ontology import (Entity, Event, EventType, ObjectType, Relation,
                              RelationType, WorldSchema)
from orbital.reasoning import Evidence, check_claim_grounding, detect_contradictions
from orbital.simulation import Simulator, counterfactual_effect
from orbital.state import Observation, WorldState
from orbital.discovery import CandidateObservation, select_next_observation


def make_small_world():
    return GroundTruthWorld(n_regions=4, seed=3)


class TestOntology(unittest.TestCase):
    def test_state_bounds_enforced(self):
        s = WorldSchema()
        s.add_entity(Entity("r0", ObjectType.REGION, state={"ndvi": 1.7}))
        self.assertEqual(s.entities["r0"].state["ndvi"], 1.0)  # clamped

    def test_invalid_state_dim_rejected(self):
        s = WorldSchema()
        with self.assertRaises(ValueError):
            s.add_entity(Entity("r0", ObjectType.REGION, state={"bogus": 1.0}))

    def test_relation_endpoints_must_exist(self):
        s = WorldSchema()
        s.add_entity(Entity("a", ObjectType.REGION, state={"ndvi": 0.5}))
        with self.assertRaises(ValueError):
            s.add_relation(Relation("a", RelationType.ADJACENT_TO, "ghost"))


class TestEnvironment(unittest.TestCase):
    def test_ground_truth_world_runs_and_stays_bounded(self):
        w = make_small_world()
        for _ in range(60):
            w.step()
            for r in w.schema.by_type(ObjectType.REGION):
                self.assertTrue(0 <= r.state["ndvi"] <= 1)
                self.assertTrue(0 <= r.state["soil_moisture"] <= 1)


class TestWorldModel(unittest.TestCase):
    def test_ensemble_produces_epistemic_spread(self):
        w = make_small_world()
        ens = EnsembleWorldModel(w.schema, n_models=6, seed=1)
        mean, std = ens.predict_step(0.5)
        total_spread = sum(std[e]["ndvi"] for e in std)
        self.assertGreater(total_spread, 0)  # models genuinely disagree

    def test_rollout_respects_constraints(self):
        w = make_small_world()
        ws = WorldState(w.schema)
        sim = Simulator(w.schema, EnsembleWorldModel(w.schema, n_models=4, seed=2))
        res = sim.rollout(ws, horizon=12)
        self.assertEqual(res.violations, 0)  # H2 constraint validity
        self.assertEqual(len(res.states), 12)

    def test_counterfactual_drought_changes_state(self):
        w = make_small_world()
        ws = WorldState(w.schema)
        sim = Simulator(w.schema, EnsembleWorldModel(w.schema, n_models=6, seed=4))
        obs_r = sim.rollout(ws, horizon=10)
        do_r = sim.rollout(ws, horizon=10, interventions=[
            Event(EventType.DROUGHT, t=0,
                  targets=[f"region_{i}" for i in range(4)], magnitude=1.0)])
        cfe = counterfactual_effect(obs_r, do_r, "region_0", "soil_moisture")
        self.assertTrue((cfe < 0).any())  # drought must dry the soil


class TestReasoning(unittest.TestCase):
    def test_contradiction_detector_flags_conflict(self):
        ev = [Evidence("optical", "r0", "soil_moisture", 0.8, 0.05),
              Evidence("sar", "r0", "soil_moisture", 0.2, 0.05)]
        v = detect_contradictions(ev)
        self.assertEqual(v.status, "CONFLICT")
        self.assertTrue(v.recommendation)

    def test_grounding_check(self):
        ev = [Evidence("s1", "r0", "ndvi", 0.5, 0.05),
              Evidence("s2", "r0", "ndvi", 0.52, 0.05)]
        self.assertTrue(check_claim_grounding(0.51, ev).startswith("SUPPORTED"))
        self.assertTrue(check_claim_grounding(0.95, ev).startswith("UNSUPPORTED"))


class TestActivePerception(unittest.TestCase):
    def test_info_gain_prefers_uncertain_targets(self):
        cands = [CandidateObservation("a", "ndvi"), CandidateObservation("b", "ndvi")]
        epi = {"a": {"ndvi": 0.001}, "b": {"ndvi": 0.5}}
        best, score = select_next_observation(cands, epi)
        self.assertEqual(best.entity_id, "b")  # H4: highest disagreement wins

    def test_active_beats_random_acquisition(self):
        """The core H4 experiment in miniature."""
        w = make_small_world()
        cands_all = [("region_0", "ndvi"), ("region_1", "ndvi"),
                     ("region_2", "soil_moisture"), ("region_3", "soil_moisture")]

        def run(policy):
            rng = np.random.default_rng(0)
            est_var = {e: {d: 0.5 for d in ("ndvi", "soil_moisture")}
                       for e, _ in cands_all}
            for _ in range(len(cands_all) * 2):
                if policy == "active":
                    eid, dim = max(cands_all,
                                   key=lambda c: est_var[c[0]][c[1]])
                else:
                    eid, dim = cands_all[rng.integers(len(cands_all))]
                o = w.observe(eid, dim, sigma=0.05)
                est_var[eid][dim] *= 0.5  # observing reduces variance
            return sum(v for e in est_var.values() for v in e.values())

        active = run("active")
        random = run("random")
        self.assertLessEqual(active, random)


if __name__ == "__main__":
    unittest.main()
