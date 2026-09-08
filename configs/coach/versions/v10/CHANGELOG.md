# v10 — generation 4 (2026-09-08)

parent: v7p
hypothesis: Raising release_min_success from 0.3 to 0.5 delays the curriculum-release phase (and its lock + auto-invariant) until the policy is reliably walking, giving the coach flexibility to temporarily lower target_ms for gait stabilization on stairs-medium-traditional (success 0.55–0.72) without triggering the lock, while leaving rough settings (success 0.98+) completely unaffected.
expected_delta_j: 0.03

## Rationale
Both rejected candidates (v8: max_params→2, v9: 1-per-3-reports) tried to reduce intervention frequency and both catastrophically hurt rough-medium-naive (one seed collapsed to J=0.117) because the early breakout requires rapid, multi-param moves. The real residual weakness is stairs-medium-traditional (J=1.251, success 0.64), where the coach's worst kept move was target_ms→1.5 at step 28M with only 52% success (effect −0.313). The release phase (and its curriculum_lock preventing target_ms reduction) currently triggers at success ≥ 0.3 — very early in the stairs run. Raising the threshold to 0.5 lets the coach use a short target_ms dip to steady the gait between 0.3 and 0.5 success before the lock engages, and delays the auto-invariant by a few reports. On rough settings (success 0.98–1.00) the threshold is trivially met, so nothing changes. This is a single guardrail parameter, no prompt edit needed.

## Code ideas (not applied)
- Log whether the release phase actually triggered in each run (first report where success ≥ release_min_success and progress ≥ release_from_progress) so we can verify the delay on stairs vs rough in the next evaluation cycle
- If the next run still shows the coach raising target_ms too aggressively on mid-success stairs (success 0.6-0.7), consider adding a soft prompt hint: 'When success is between 0.5 and 0.8, prefer raising target_ms by no more than 0.2 m/s per intervention to let the gait adapt.'
