# Experiment Plan V1

## Can Run Now
- v1 core matrix (static + heuristic + TriSatFlow checkpoint baseline)
- mobility-aware main and mobility-stress visible profile comparison
- architecture ablation over `only_leo/leo_geo/leo_ground/full`
- paper table/figure-data export

## Needs Mobility-Aware Profile Improvement First
- claims on mobility-safe profile as fully production-equivalent training profile
- strict conclusions about mobility-risk reduction causality

## Depends on Full HMADRL Loop
- final HMADRL-vs-TriSatFlow learning-curve comparison
- fairness checks involving fully-trained HMADRL checkpoints

## Main vs Optional Metrics
- main: task success, delay, mobility/deadline failure, regret, execution reliability
- optional: energy (requires manual audit)

## Why Energy Is Not Main Yet
- SatEdgeSim energy counter semantics are still under audit
- unit/aggregation interpretation must be finalized before headline claims
