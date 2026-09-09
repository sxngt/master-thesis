# v12 — generation 3 (2026-09-09)

parent: v7p
hypothesis: Raising noise_z from 2.0 to 2.5 inflates the reported noise threshold, causing the small coach model to judge fewer (especially weak) trends as significant on stairs, reducing the 11.7 interventions (2 rollbacks, 0.7 restores) to a smaller set of higher-conviction moves, while leaving the strong-signal rough settings (where the trend is well above 2.5σ) essentially untouched.
expected_delta_j: 0.08

## Rationale
The incumbent's dJ sign accuracy is 0.43 (worse than coin-flip) with a +0.052 over-optimism bias, and the worst interventions are all on the noisy stairs setting where the coach acts on weak trends and gets rolled back (2 rollbacks, 0.7 restores). Both rejected candidates (v10, v11) added prompt complexity and catastrophically broke rough-medium-naive s1 (J=0.117, 16 thrashing moves), confirming the small model is very sensitive to extra rules. A settings-only change to noise_z avoids all prompt risk: it simply raises the reported noise floor the coach reads, making it more conservative about what counts as a real signal. On rough settings the J trend is consistently 5-20× the noise, so a 25% increase in the threshold is immaterial. On stairs, where per-report J swings are within 1-2σ, the higher threshold will cause the coach to say 'no change' more often, saving it from the 2 rollbacks and giving PPO more uninterrupted budget to improve the gait. This is the cleanest testable single change: fewer interventions on the noisy setting, same interventions on the clean settings.

## Code ideas (not applied)
- Log the number of 'significant beyond noise' vs 'within noise' judgments the coach makes per setting, to confirm that noise_z=2.5 actually reduces interventions on stairs but not on rough.
- Track whether the reduced intervention count on stairs correlates with a higher final J, or whether the saved budget is simply unused (i.e., the coach just trains without ever raising target_ms on stairs).
