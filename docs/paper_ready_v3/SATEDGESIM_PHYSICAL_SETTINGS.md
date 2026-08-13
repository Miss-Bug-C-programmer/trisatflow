# SatEdgeSim Paper V3 Physical Settings

Stage 8 uses an isolated SatEdgeSim settings root:

```text
SatEdgeSim/settings/paper_v3_actual/
```

It is a copy of the upstream default settings with only the paper actual-bank physical envelope changed. The default root remains untouched.

## Calibration

- `edge_devices_range=8000000`
- `edge_datacenters_coverage=8000000`
- `cloud_coverage=20000000`
- `simulation_time=60`
- `tasks_generation_rate=8`
- charts disabled and `pause_length=0` for headless REST execution

The default SatEdgeSim root uses 32,000-40,000 km radio/coverage ranges, which makes most abstract tiers visible for long spans and can collapse exported masks toward `(1,1,1,1)`. The paper root uses mid-scale LEO/ground footprints instead of global coverage so topology transitions are visible in actual traces while still keeping enough candidates for `paper_strict` receipt and replay gates. The ranges are intentionally wider than a strict single-hop horizon because this SatEdgeSim topology model maps edge, GEO/cloud, and ground resources through discrete location files rather than a full orbital propagator.

Exporter reset arguments remain authoritative for run-specific `devices_count`, seed, source mode, scenario profile, and decision budget. Formal manifests record the selected settings root and SHA-256 hashes for all required input files.

## Live Preflight Contract

Stage 13 separates local TriSatFlow audits from live SatEdgeSim audits:

- `preflight-offline` does not start SatEdgeSim and only checks local contracts, trace-bank manifests, and reporting inputs.
- `preflight-satedgesim` requires a running REST bridge, Maven compile from `SATEDGESIM_ROOT`, `/version` provenance, receipt checks, and lower-action binding audit.
- `build-traces` is the only pipeline mode that generates actual and controlled-stress paper trace banks.

The TriSatFlow repository must not hardcode the temporary upstream checkout path. Use `SATEDGESIM_ROOT` or `--satedgesim-root` when running live modes.
