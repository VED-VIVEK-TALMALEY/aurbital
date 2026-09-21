"""ORBITAL — World-State Intelligence & Simulation Engine.

A research system converting heterogeneous observations of the physical world
into a persistent, causal, executable world-state model.

Layers (see docs/research_problem.md):
    ontology      typed objects/relations/events/actions (schema of reality)
    state         persistent world state + estimator
    models        transition dynamics, ensembles, uncertainty
    simulation    forward rollouts, counterfactuals
    discovery     information-gain active perception, hypotheses
    reasoning     contradiction detection, LLM interface (replaceable)
"""

__version__ = "0.1.0"
