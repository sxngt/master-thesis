# v7 — generation 2 (2026-09-08)

parent: v5
hypothesis: Adding an explicit rule 'never lower target_ms when the J trend is positive over 2+ reports' prevents the most common harmful pattern (lowering curriculum during a breakthrough) seen in 4 of the 6 worst v5 interventions, raising J on stairs and rough-hard without disturbing the already-stuck rough-medium-naive.
expected_delta_j: 0.08

## Rationale
In v5, 4 of the 6 worst-intervention effects are target_ms reductions during positive trends (rough-hard s0 step 20M target→0.9, rough-hard s2 step 22M target→0.9, stairs s0 step 25M target→0.85, plus stairs gait changes at a similar 'breakthrough' moment). The existing prompt already states a reduced target 'forfeits the velocity term' yet the coach still does it—likely because it reads a J jump as fragility and 'consolidates' by lowering the curriculum. v6's stuck-protocol made things worse by increasing intervention count on stairs (13 kept vs 6) while fixing nothing on rough-medium. A single, unambiguous prohibition on lowering target_ms during a rising trend removes this specific failure mode without adding any new decision protocol, keeps the coach's intervention count on the healthy settings at the v5 level (~6), and cannot hurt rough-medium (where target_ms is already at 0.3-1.05 and the problem is 0% success, not curriculum level).

## Code ideas (not applied)
- Log a flag 'target_ms_lowered_during_positive_trend' in the intervention ledger so future coach versions can see at a glance how often this forbidden pattern occurs and whether the rule actually stops it.
- In the evidence section, add a one-line 'curriculum direction' hint computed from the J trace: 'trend: rising → only raise target_ms' or 'trend: flat/falling → curriculum flexible'. This gives the small model a pre-computed decision rather than requiring it to infer from the trace.
