from __future__ import annotations

from trisatflow.envs.physical_metrics import ENERGY_CONVERSION_RULE_WH_TO_J, energy_delta_from_cumulative_wh


def test_cumulative_wh_counter_converts_to_step_joules() -> None:
    payload = energy_delta_from_cumulative_wh(10.5, 10.0)

    assert payload["raw_energy_counter_wh"] == 10.5
    assert payload["previous_raw_energy_counter_wh"] == 10.0
    assert payload["step_energy_delta_wh"] == 0.5
    assert payload["step_energy_delta_j"] == 1800.0
    assert payload["energy_conversion_rule"] == ENERGY_CONVERSION_RULE_WH_TO_J


def test_cumulative_wh_counter_delta_is_not_negative() -> None:
    payload = energy_delta_from_cumulative_wh(9.5, 10.0)

    assert payload["step_energy_delta_wh"] == 0.0
    assert payload["step_energy_delta_j"] == 0.0
