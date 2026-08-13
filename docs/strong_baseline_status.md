# Strong Baseline Status

This repair adds runnable strong-baseline machinery for CPU smoke validation. The tiny results are engineering checks only and are not paper results.

| baseline | status | update/train step | mask support | continuous action support | paper_ready |
|---|---|---:|---:|---:|---:|
| `pdqn_hybrid` | implemented for tiny training and checkpoint save/load | yes | yes | yes | false |
| `flat_hybrid_ac` | implemented for tiny training and checkpoint save/load | yes | yes | yes | false |
| `attention_candidate` | candidate-level attention forward/select only | no full RL update | yes | yes | false |
| `small_scale_grid_oracle` | implemented as exact grid when small, beam approximation when large | not trainable | yes | yes | false |

## Claim Boundary

Allowed now:

> We implemented trainable P-DQN-style and flat hybrid actor-critic baselines with CPU tiny update tests, and a small-scale grid oracle for oracle-gap diagnostics.

Not allowed now:

> TriSatFlow significantly outperforms strong hybrid baselines.

That claim requires full multi-seed GPU training/evaluation under the same observation, mask, and lower-action fairness conditions.

## Table Readiness

The trainable baselines can enter a future formal Table only after:

- full multi-seed training is run;
- action mask source and observation policy match TriSatFlow;
- lower continuous resource semantics are reported consistently;
- checkpoint selection and statistical testing use training seed/checkpoint as the primary statistical unit;
- `paper_ready` is changed only by a full experiment report, not by smoke scripts.

