# v11 — generation 2 (2026-09-09)

parent: v7p
hypothesis: Adding a concrete raise-rate rule (block target_ms increases when tracking ratio < 0.80 AND success < 0.85) prevents the premature speed raises that cause costly rollbacks on stairs-medium-traditional, letting the policy stabilise its gait first, while leaving rough settings (where both conditions are already met) completely unaffected.
expected_delta_j: 0.08

## Rationale
The incumbent's stairs-medium-traditional is the weakest setting (J=0.913, success 0.42). The coach's worst interventions on all three settings are raising forward_velocity.target_ms to 1.3–1.5 while the tracking ratio is only 0.65–0.81 and success is 40–55 %, causing rollbacks that undo 2 M+ steps of learning. The existing prompt says 'a higher target buys speed once the gait is reliable' but gives no numeric threshold, so the small local model (dJ sign accuracy 0.43) routinely overrides the intent and maxes out the command. Rough settings already have success > 0.97 and tracking > 0.87, so the new rule is inactive there. One concrete, numeric rule in the curriculum section is the smallest edit that removes the demonstrated failure without touching anything that works.

## Code ideas (not applied)
- Add a guardrail that clips any proposed forward_velocity.target_ms increase to +0.10 m/s when the last deterministic eval has tracking_ratio < 0.80 and success < 0.85, mirroring the prompt rule in code so it cannot be ignored by the model.
- Track and log the tracking ratio and success at each intervention in the report so the coach (and the evaluator) can verify the raise-rate rule was respected.
