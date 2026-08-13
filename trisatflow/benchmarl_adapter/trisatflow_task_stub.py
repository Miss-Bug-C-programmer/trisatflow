"""Backward-compatible import shim for experimental TorchRL adapters."""

from trisatflow.benchmarl_adapter.torchrl_env import TriSatFlowBenchMARLEnv, TriSatFlowTorchRLEnv

__all__ = ["TriSatFlowTorchRLEnv", "TriSatFlowBenchMARLEnv"]
