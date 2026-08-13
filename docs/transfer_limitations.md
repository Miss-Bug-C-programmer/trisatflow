# Transfer Limitations Audit

This document records the current constellation-size transfer boundary. It is a blocker inventory, not evidence that 16-to-32/64 transfer works.

| module | fixed_size_dependency | required_refactor | current_status |
|---|---|---|---|
| lower centralized critic | Critic inputs commonly flatten all LEO embeddings/actions into checkpoint-shaped vectors. | Replace flatten critic with permutation-invariant pooling or graph-level readout. | transfer_blocked_by_fixed_size_module_for_checkpoint_16_to_32_64 |
| QMIX/VDN mixing networks | Mixers are initialized with a fixed `n_agents` dimension. | Use agent-count invariant mixer or per-agent pooling adapter. | checkpoint_transfer_not_claimed |
| replay buffer tensors | Stored transitions are shaped by the training `n_leo`. | Do not mix train and transfer replay batches without padding/masking. | evaluation_reset_only_safe |
| obs_builder candidate features | Per-node observations are variable-size, but downstream flatten consumers may be fixed. | Keep per-node tensors until pooled policy/critic readout. | reset_shape_check_required |
| graph encoder pooling | Message passing can be variable-size if pooled before fixed heads. | Verify checkpoint heads consume pooled or shared per-node features. | not_sufficient_for_transfer_claim_alone |
| upper action head | Shared per-agent action head is reset-compatible, but checkpoint compatibility depends on encoder/critic. | Shape-test checkpoint forward on 32/64 before transfer claims. | inductive_transfer_unproven |

## Safe Claim Boundary

Current CPU stress scripts may show that the analytic environment can reset and step for larger `n_leo` under rule/random policies. That is not a trained checkpoint transfer result.

Allowed wording:

> We provide a stress/transfer audit harness and report reset/step compatibility separately from checkpoint transfer compatibility.

Forbidden without successful checkpoint forward/evaluation on 32/64:

> The GNN policy supports constellation-size transfer from 16 to 32/64 satellites.

