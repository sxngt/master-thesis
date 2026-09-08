# v10 — generation 5 (2026-09-08)

parent: v7
hypothesis: Extending the CRITICAL RULE from 'do not lower target_ms during a positive J trend' to 'do not change ANY parameter during a positive J trend (empty actions or raise target_ms only)' will eliminate the 4-5 demonstrated harmful interventions per run where the coach reshapes reward weights mid-breakthrough, raising J especially on rough-medium-naive and stairs-medium-traditional.
expected_delta_j: 0.15

## Rationale
The worst-intervention list shows 5 of 6 negative-effect changes are 'change a param while J is rising' (target_ms lowered, stability/alive_bonus/feet_air_time weights altered during a 3-5 report climb). The current CRITICAL RULE only protects target_ms from being lowered; it does not stop the coach from editing reward weights during a breakthrough. The coach's dJ sign accuracy is 0.51 (random), so its 'insight' to re-shape weights mid-rally is essentially noise minus the momentum lost. v7's +0.808 gain on rough-medium-naive came from the target_ms rule; extending the same logic to all parameters should prevent the remaining harmful edits without removing the curriculum-raising behaviour that helped rough-hard-traditional (all 3 seeds end at target_ms 1.50). This is a single, isolated rule change in system_md; no settings touched.

## Code ideas (not applied)
- Add a hard gate in the action-applying code: if the last 2+ objective trace points are strictly increasing, reject any action that is not 'raise forward_velocity.target_ms' with an automatic no-op and log the rejection. This enforces the CRITICAL RULE regardless of LLM compliance.
- Track per-run 'violations' count (times the coach changed a non-target_ms param during a rising trend) and report it in the next evidence section so the coach sees its own compliance rate.
