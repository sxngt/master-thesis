# v8 — generation 3 (2026-09-08)

parent: v7 (excluded from the evolve3 queue 2026-09-08 — rough dJ −0.065 under the old PPO, docs/coach_versions.md §6.4)
hypothesis: Limiting each intervention to 1 parameter (max_params 3→1) prevents the coordinated 2-param reward changes that are consistently the most harmful (all 4 two-param changes in the worst-intervention list have negative detrended effects), while the successful rough-hard runs already operate with single-param curriculum moves, so the change mainly removes the demonstrated failure mode on stairs and rough-medium-naive.
expected_delta_j: 0.08

## Rationale
The 6 worst detrended interventions split cleanly: the 2-param changes (stability+feet_air_time, alive_bonus+stability, action_rate+alive_bonus) all hurt J (-0.265 to -0.152) and were kept (not rolled back because J was near floor). The 2 single-param changes (lowering target_ms) are also harmful but are now caught by the CRITICAL RULE added in v7. The successful rough-hard seeds (J≈1.63) used mostly single-param target_ms moves. On the failed rough-medium-naive seed 0, 13 kept interventions with up to 3 params each means up to 39 parameter perturbations that compound and make the ledger uninformative. Forcing one param per step halves the perturbation budget, sharpens attribution in the evidence ledger, and structurally prevents the 'lower stability AND raise air-time' packages that the coach's dJ predictions (sign accuracy 0.51) consistently mis-score as positive.

## Code ideas (not applied)
- Log the number of params per intervention in the KPI table so we can confirm the single-param constraint is actually binding (i.e., the model would have proposed 2+ on some steps).
- If max_params=1 proves too restrictive (e.g., rough-medium-naive seed 0 still stuck), consider adding a 'pair' exception: allow 2 params only if they are the same reward component's weight AND target (e.g., energy.weight + energy.target), which is a semantically coordinated adjustment rather than two independent penalties.
