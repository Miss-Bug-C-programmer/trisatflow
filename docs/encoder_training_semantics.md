# Encoder Training Semantics

This audit separates action collection detach/no_grad from training-update
detach. Action collection may use `torch.no_grad()` and detached embeddings for
environment interaction without implying that lower training updates also detach
the encoder.

## Modes

- `shared_upper_only`: compatibility default. The upper actor-critic trains the
  shared encoder. During lower updates, the lower allocator consumes the shared
  embedding as a fixed representation and does not update the shared encoder.
- `shared_joint`: upper and lower updates both allow gradients into the shared
  encoder. This claim is valid only when the gradient diagnostic reports a
  positive `shared_encoder_grad_norm_from_lower`.
- `separate_lower_encoder`: the lower allocator owns a separate encoder and
  target encoder. Lower gradients update the separate lower encoder, not the
  shared upper encoder.

Legacy aliases remain accepted:

- `shared_frozen` -> `shared_upper_only`
- `separate` -> `separate_lower_encoder`

## Current Tiny Smoke Result

The CPU smoke verifies:

- `shared_upper_only`: lower shared-encoder grad is zero.
- `shared_joint`: lower shared-encoder grad is positive.
- lower action is structurally conditioned on upper action via
  `node_embedding + one_hot(upper_action)`.
- action-collection detach does not imply training-update detach.

## Credit Assignment

The lower actor input schema is:

`node_embedding + one_hot(upper_action)`

The MADDPG lower critic input schema is:

`flatten(all_agent_embedding, upper_action_onehot, lower_action)`

Thus lower continuous allocation is explicitly conditioned on the upper
discrete offloading action. The current centralized critic is initialized with
the active environment's number of LEO agents; do not describe it as
agent-count invariant unless a pooling critic is introduced and tested.

## Safe Paper Wording

If using `shared_upper_only`:

the encoder is learned by the upper actor-critic and reused as a fixed
representation by the lower allocator during lower updates.

Only if `shared_joint` passes the gradient diagnostic may the paper state:

shared encoder is jointly trained by both decision levels.
