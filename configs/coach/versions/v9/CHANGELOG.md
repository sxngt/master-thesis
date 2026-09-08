# v9 — generation 4 (2026-09-08)

parent: v7 (excluded from the evolve3 queue 2026-09-08 — stairs dJ −0.127 ± 0.050 with the 3M interval, docs/coach_versions.md §6.4)
hypothesis: Increasing the coach interval from 2M to 3M steps reduces over-intervention (the dominant failure mode: 10-14 kept changes per run, with 5 of 6 worst interventions occurring during positive trends where the coach should have waited), giving the policy more time to converge per change and cutting the harmful thrash cycle on rough-medium-naive.
expected_delta_j: 0.08

## Rationale
The worst-intervention list is dominated by (a) lowering target_ms during a rising J trend (violating the CRITICAL RULE already in the prompt) and (b) 2-param reward tweaks during 0%-success explore phases. v8 showed that restricting param count (max_params=1) backfired—the coach compensated with MORE interventions (17-18) and worse sign accuracy (0.39). The small local model does not reliably follow 'prefer no change' instructions, so a structural timing constraint is more robust than another prompt rule. At 3M intervals the coach gets ~13 reports over 40M steps instead of ~20; the critical target_ms raises on rough-hard (already at 1.50 in all seeds) are captured with fewer reports because they are large discrete jumps, while the harmful micro-tweaks during explore are simply skipped. The curriculum-release invariant still steps target_ms up automatically, so no curriculum regression is expected.

## Code ideas (not applied)
- Log a per-intervention 'time_since_last_change' feature so we can confirm in future diagnostics that the interval change actually reduced harmful mid-trend actions rather than just shifting them to different steps.
