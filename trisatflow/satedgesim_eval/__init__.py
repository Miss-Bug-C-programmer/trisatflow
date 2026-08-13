"""Evaluation-only bridge from a frozen TriSatFlow policy to SatEdgeSim REST.

This package is intentionally not a training environment.  It only loads a
saved checkpoint, performs deterministic policy inference, maps the abstract
TriSatFlow action into SatEdgeSim's concrete VM index, and sends the action to
SatEdgeSim through the REST API.
"""

from trisatflow.satedgesim_eval.client import SatEdgeSimClient
from trisatflow.satedgesim_eval.frozen_policy import FrozenTriSatFlowPolicy

__all__ = ["SatEdgeSimClient", "FrozenTriSatFlowPolicy"]
