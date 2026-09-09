# v13 — generation 4 (2026-09-09)

parent: v7p
hypothesis: Reducing max_params from 3 to 2 forces the coach to make more focused, single- or dual-parameter interventions, reducing multi-param interactions that are hard to attribute and increasing the chance that harmful compound moves (especially on stairs where 11.7 interventions include 3-param changes) are avoided, while having minimal impact on rough settings where the coach already succeeds with 1-2 param moves.
expected_delta_j: 0.06

## Rationale
The incumbent's worst interventions include 3-param changes (e.g. v12 rough-medium s1 at 2M: target_ms + energy.weight + action_rate.weight; at 14M: alive_bonus + termination + target_ms). On stairs-medium-traditional the coach makes 11.7 interventions with 2.0 rollbacks and a dJ sign accuracy of only 43%—the small model struggles to attribute multi-param changes and compounds errors. Stairs s2 (J=0.509, success=0.20) shows the policy collapsing under 12 interventions with 3 rollbacks, consistent with multi-param confusion. Rough settings (0-1 rollbacks, J~1.65-1.69) already succeed with 1-2 param moves per intervention, so capping at 2 should not reduce their coaching quality. This is a single guardrail change, easy to attribute, and directly targets the demonstrated failure mode of multi-param interactions on the weakest setting.

## Code ideas (not applied)
- (none)
