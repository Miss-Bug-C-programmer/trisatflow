# Statistical Protocol Report

Training seed/checkpoint is the primary independent statistical unit.

Offline test seeds are repeated evaluations of the same checkpoint and must be
aggregated within each method x train_seed/checkpoint before pairwise tests.
Online replay seeds are repeated SatEdgeSim replay observations within a
checkpoint cluster unless each online seed comes from an independently trained
checkpoint.

## Required Outputs

The statistical protocol writes:

- `method_summary.csv`: method-level checkpoint summaries with bootstrap CI.
- `pairwise_tests.csv`: paired checkpoint-level tests with paired t-test,
  Wilcoxon signed-rank test, Holm-corrected p-values, Cohen's dz, Cliff's delta,
  and cluster bootstrap CI.
- `claim_guard.json`: machine-readable downgrade rules for paper claims.
- `statistical_protocol_report.md`: this protocol summary.

Every pairwise row reports `n_rows`, `n_effective_pairs`, `n_train_seeds`,
`n_checkpoints`, and `statistical_unit`. `n_rows` is descriptive only; statistical
inference is based on `n_effective_pairs`.

## Claim Guard

When the best mean method and runner-up are not Holm-significant, the allowed
claim is:

`mean-ranked reference only; statistically comparable`

Required Table 3 wording:

IPPO+MADDPG is selected as a mean-ranked reference pairing; the four pairings are
statistically comparable under Holm-corrected pairwise tests.

Forbidden wording unless supported by the reported protocol:

- significantly outperforms
- clearly best
- statistically superior
- dominates

If `n_effective_pairs < 5`, `small_n_warning=true` is required and the paper must
use cautious wording even if a p-value is small.
