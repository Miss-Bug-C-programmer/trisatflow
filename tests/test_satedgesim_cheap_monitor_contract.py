from __future__ import annotations

from trisatflow.adapters.backend import BackendCapabilities
from trisatflow.adapters.satedgesim_client import SatEdgeSimBackend


def test_cheap_monitor_parser_preserves_truthful_optional_fields() -> None:
    backend = object.__new__(SatEdgeSimBackend)
    backend._configuration = None
    backend._capabilities = BackendCapabilities(topology_source="TopologyOracle")

    monitor = backend._monitor_from_payload(
        {
            "payloadKind": "cheap_monitor",
            "simulationTimeSec": 12.0,
            "configId": "cfg-7",
            "configVersion": 3,
            "configurationAgeSec": 4.5,
            "serviceRateObserved": 125000.0,
            "serviceRateLowerBound": 125000.0,
            "serviceHorizonSec": 30.0,
            "queueSummary": {"arrivedTaskCount": 2, "unfinishedTaskCount": 1},
            "remainingWorkload": {"total": 75, "source:1": 75},
            "deadlineSlack": {"42": 8},
            "instrumentation": {
                "candidateEvaluations": 0,
                "fullStateBuilderInvoked": False,
            },
            "containsFutureStochasticState": False,
        },
        source="/get_monitor_state",
        true_cheap=True,
    )

    assert monitor.acquisition.cheap_monitor_verified is True
    assert monitor.current_config_id == "cfg-7"
    assert monitor.current_config_version == 3
    assert monitor.configuration_age_sec == 4.5
    assert monitor.remaining_workload_summary == {"total": 75.0, "source:1": 75.0}
    assert monitor.service_rate_observed == 125000.0
    assert monitor.service_rate_lower_bound is None
    assert monitor.service_bound_certified is False
    assert monitor.service_horizon_sec == 30.0
    assert monitor.metadata["service_rate_observed_available"] is True
    assert monitor.metadata["service_rate_lower_bound_available"] is False


def test_cheap_monitor_parser_does_not_invent_uncertainty_or_capacity() -> None:
    backend = object.__new__(SatEdgeSimBackend)
    backend._configuration = None
    backend._capabilities = BackendCapabilities()

    monitor = backend._monitor_from_payload(
        {
            "payloadKind": "cheap_monitor",
            "simulationTimeSec": 0.0,
            "remainingWorkload": {"total": 0},
            "instrumentation": {
                "candidateEvaluations": 0,
                "fullStateBuilderInvoked": False,
            },
            "containsFutureStochasticState": False,
        },
        source="/get_monitor_state",
        true_cheap=True,
    )

    assert monitor.prediction_uncertainty == {}
    assert monitor.local_load_summary == {}
    assert monitor.service_rate_lower_bound is None
    assert monitor.service_horizon_sec is None


def test_cheap_monitor_parser_requires_explicit_uncertainty_evidence_marker() -> None:
    backend = object.__new__(SatEdgeSimBackend)
    backend._configuration = None
    backend._capabilities = BackendCapabilities()
    payload = {
        "payloadKind": "cheap_monitor",
        "predictionUncertainty": {"service": 0.0},
        "instrumentation": {
            "candidateEvaluations": 0,
            "fullStateBuilderInvoked": False,
            "predictionUncertaintyAvailable": False,
        },
        "containsFutureStochasticState": False,
    }
    unmarked = backend._monitor_from_payload(payload, source="/get_monitor_state", true_cheap=True)
    assert unmarked.prediction_uncertainty == {"service": 0.0}
    assert unmarked.uncertainty_evidence_available is False

    payload["uncertaintyEvidenceAvailable"] = True
    marked = backend._monitor_from_payload(payload, source="/get_monitor_state", true_cheap=True)
    assert marked.uncertainty_evidence_available is True
