# v6 — generation 1 (2026-09-08)

parent: v5
hypothesis: Adding a 'stuck protocol' that mandates a single decisive reward simplification when success_rate is 0% for 5+ reports will break the thrashing loop on rough-medium-naive (15-17 unhelpful kept interventions, 0% success) and raise its J toward the ~1.1 level of the other settings, improving mean J.
expected_delta_j: 0.45

## Rationale
The evidence is unambiguous: rough-medium-naive produces 15-17 kept interventions with 0 rollbacks yet success stays at 0.00 and velocity at 0.20-0.23 m/s. The coach is over-intervening (3× the kept count of the working rough-hard setting) but every change passes the rollback test because J is already at the noise floor (~0.11). The system prompt explicitly tells the coach 'that plateau is not a collapse,' so it keeps making incremental 1-2 param tweaks with overconfident positive dJ predictions (sign accuracy 0.42, bias +0.031). On the working setting the coach converges in 5-6 moves; on the broken one it thrashes 15-17 moves across 15+ parameters (target_ms ends at 1.05, 0.41, and 0.30 across seeds — no consistent curriculum). The fix is a concrete 'stuck protocol': after 5 reports at 0% success, stop tweaking and make one structural reset (target_ms=0.3, keep only the single largest penalty, zero the rest) to create a single clear gradient. This is directly testable: if rough-medium-naive's J rises while the other two settings stay stable, the paired t-test will confirm.

## Code ideas (not applied)
- Add a 'stuck counter' to the decision layer: track consecutive reports with success_rate < 5%. Expose it as a field in the user template (e.g. {stuck_count}) so the coach can see the number explicitly rather than counting from the trace.
- If stuck_count >= 3, automatically clamp max_params to 1 and max_rel_change to 0.1 to force the coach to make a single minimal move, reducing thrashing even if the LLM ignores the prompt instruction.
- Log the coach's target_ms trajectory per seed. The 1.05/0.41/0.30 spread across rough-medium-naive seeds suggests no coherent curriculum; a monotonicity check (warn if target_ms oscillates by >2x between consecutive interventions) could flag incoherent curriculum strategies.
