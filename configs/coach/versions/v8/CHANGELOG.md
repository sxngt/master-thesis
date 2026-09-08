# v8 — generation 2 (2026-09-08)

parent: v7p
hypothesis: Capping simultaneous parameter changes at 2 (from 3) will improve attribution quality and reduce destabilization, particularly on stairs-medium-traditional where the coach currently makes 14-17 scattered changes including a rolled-back 3-param move, while rough settings that already make mostly 1-param moves are unaffected.
expected_delta_j: 0.05

## Rationale
Stairs-medium-traditional is the weak setting (J=1.251, success 0.64) and the coach makes the most kept moves (14-17) with the poorest outcome per change. Its single rollback was a 3-param simultaneous move (stability.weight, alive_bonus.weight, termination.weight at step 4M) that destabilized the policy. Meanwhile, rough-hard and rough-medium already make mostly single-param changes (target_ms adjustments) and achieve >0.98 success with 0 rollbacks, so restricting max_params to 2 removes the 3-param failure mode without harming the settings that are already working. Fewer simultaneous changes also improves the coach's dJ attribution (currently 0.55 sign accuracy) by making each move's effect easier to isolate from the learning trend, which should lead to better subsequent decisions on the high-variance stairs setting.

## Code ideas (not applied)
- (none)

## Note (2026-09-09)
The driver wrote only the meta-LLM's delta into coach.yaml and dropped the
parent's (v7p) overrides — playbook, settled_reports, ledger_veto_obs,
curriculum_lock — so the evolve3 runs of this version measured v7p minus its
decision layer plus the hypothesis (docs/coach_versions.md §6.12). The
verdict does not test the hypothesis. Fixed in evolve.py Proposal.resolve.
