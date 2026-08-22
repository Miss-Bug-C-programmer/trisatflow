# SatEdgeSim backend contract v2

TriSatFlow consumes SatEdgeSim only through the declared `/capabilities`
contract. The formal planner acquisition call is:

```json
{
  "scope": {"task_ids": [], "source_ids": [], "node_ids": [], "link_ids": [], "route_ids": [], "resource_keys": []},
  "budget": {"max_candidate_count": 32, "max_planner_evaluations": 32},
  "fidelityHint": "light"
}
```

The adapter never chooses between old `/get_planner_state_scoped` and
`/get_planner_state_budgeted`; those are compatibility routes only. A formal
capability declaration is read first. `GET /get_monitor_state` is accepted as
a true cheap monitor only when its payload proves `payloadKind=cheap_monitor`,
zero candidate evaluations, no full-state builder invocation and no future
stochastic truth.

Planner responses carry requested/applied scope and budget, before/after entity
counts, acquisition instrumentation and `postFilterOnly=false`. The adapter
does not apply a second `restrict_count` after the backend response.

Persistent configurations are sent through `/configuration/apply` and
`/configuration/dispatch`; `apply_action` is not relabelled as persistent
execution. The Java backend matches reusable selector rules for later tasks.
When a planner candidate contains no source identity (as with a destination-
only VM candidate), the canonical planner emits a source-agnostic `default`
reusable rule; it must not encode the destination device id as a future-task
source or node selector.

Selective interventions use the versioned `POST /configuration/patch` contract
when the capability declaration contains `supportsConfigurationPatch=true`.
The adapter sends the base configuration version, observed world/control
identity, optional post-delay revalidation world token and acquisition epoch,
the set-valued `requestedScope`, separate observation scope, exact
assignment/resource/route/priority/rule deltas, explicit
preserve/resume/recompute semantics, planning-delay metadata and acquisition
metadata. An authoritative backend without that
capability fails closed; the non-authoritative legacy fallback is marked as a
compatibility receipt and is not publication evidence.

The canonical intervention order is:

```text
monitor at t -> candidate/scope -> scoped acquisition -> planner
-> measured/modelled delay -> /advance_world -> post-delay /configuration/validate
-> /configuration/patch -> actual evidence
```

`baseWorldVersion` is the observation identity.  A world that evolved during
the physical delay is accepted only when the canonical validation endpoint
returns the current `worldVersion`, which is then sent as
`revalidatedWorldVersion`; a patch cannot silently apply against either an
unvalidated or stale world.

Patch application evidence is read from the structured receipt, including
requested and realized scopes, rejected changes, configuration versions,
simulation time, actual changed entities and realized reconfiguration-volume
primitives. Estimated reconfiguration cost is never copied into realized cost
when those measurements are absent.

Physical delay is accepted only when before/after backend time verifies the
requested delta. The adapter clears its time cache after `/advance_world` so a
receipt is not reused as proof of later world evolution. Missing contact
enforcement remains a hard capability boundary, not a paper claim.
