# v9 — generation 3 (2026-09-08)

parent: v7p
hypothesis: Capping interventions to one per three reports while 0.2 ≤ success < 0.8 will let the partially-walking policy consolidate its gait between changes, reducing the oscillation that is stalling stairs-medium-traditional (14-17 changes → ~7-9) while leaving rough settings (success 0.98+) unaffected because the rule is inactive above 0.8.
expected_delta_j: 0.04

## Rationale
The incumbent's weakest setting is stairs-medium-traditional (J=1.251, success 0.55-0.72) where the coach makes 14-17 interventions over 20 reports (~1.5/report), including one 3-param rollback. On rough, the coach makes 8-11 interventions and success is 0.98+. The v8 rejection (max_params 3→2) proved that spreading changes across more reports is harmful on the floor, but the opposite problem — too many changes in the 0.2-0.8 success band — is stalling stairs. The coach's dJ sign accuracy is only 55%, so half its interventions are noise; each one resets a gait that was beginning to stabilize. Rough is unaffected because success is already >0.9 throughout (except a brief <0.2 floor window before breakout, where the playbook recipe already governs). A 3-sentence rule added to the system prompt enforces the frequency cap in exactly the regime where it helps, with no settings change and no risk to the rough-medium breakout.

## Code ideas (not applied)
- (none)
