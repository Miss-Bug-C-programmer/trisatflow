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

Physical delay is accepted only when before/after backend time verifies the
requested delta. The adapter clears its time cache after `/advance_world` so a
receipt is not reused as proof of later world evolution. Missing contact
enforcement remains a hard capability boundary, not a paper claim.
