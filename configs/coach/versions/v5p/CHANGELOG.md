# v5p — v5 + evidence enforcement: playbook, settled ledger, ledger veto, curriculum lock (hand-written, 2026-09-08)

parent: v5
hypothesis: computed cross-run evidence (which moves the runs that left the success floor
applied at their first report, which moves only stuck runs made, cross-run ledger while
walking) lets the local coach reproduce the breakout recipe on rough-medium-naive instead of
leaving 2/3 seeds on the floor (evolve v5, 2026-09-08: J 0.12 for 40M steps), and stops the
curriculum wandering (target_ms lowered again while stuck) — J +0.5 on that setting; and
evidence beyond the next report (the settled ledger, J two reports after each curriculum
move) stops the coach from refusing to release the command once walking — +0.05 on rough
and stairs.
expected_delta_j: 0.15

## Rationale
Old-stack evolve v5 vs control (paired, n=9): dJ +0.44, p=0.034, but rough-medium-naive
0.643+-0.902 — only seed 0 escaped the floor. Every earlier run of that task that escaped
(16/38 across coach_v3/v4/evolve v5) had applied {target_ms down, energy.weight up,
action_rate.weight up} at the first report; the stuck local-coach seeds applied only the
target, then raised it while stuck and drifted into stability/LR moves. The coach's evidence
was in-run only; llm_feedback/playbook.py adds the index of finished runs, and system.md gets
one paragraph on how to weigh it. Prompts otherwise identical to v5 (user.md unchanged).

Settled ledger (added 2026-09-08 morning, after the new-stack v6 rough runs): the immediate
ledger row of a raised command is systematically negative — success dips at the next report
and recovers afterwards — so the v6 s1 coach held target_ms at 1.0-1.1 from 28M on
("target_ms up -0.104 +- 0.021 (6 obs)" while J had risen 1.407 -> 1.477 after 1.0 -> 1.1).
`coach.settled_reports: 2` adds a second ledger read two reports after each kept curriculum
move, net of the pre-move trend, and system.md tells the coach to weigh that row instead.

Ledger veto (added after the v6 stairs runs, same morning): v6 stairs s1 strengthened the
stability/termination penalties at 6 of its 14 reports while its own ledger read
"stability.weight down: -0.085 +- 0.006 (5 obs)", and never left the floor (J 0.245 vs v5
1.163). `coach.ledger_veto_obs: 3` drops a reward-weight move whose direction the in-run
ledger scores negative (mean + 2 sd < 0) over >= 3 observations; the history line shows the
veto so the model learns. Curriculum levers are exempt (their immediate row is biased).
Guardrail notes ("dropped") are now shown in the history for every version.

## Code ideas (not applied)
- (none)

Curriculum lock (added after v7 was proposed, same morning): v6 stairs s0 lowered target_ms
0.98 -> 0.8 at 34M with success 0.55 and the release invariant had to step it back up. The
meta-LLM's v7 forbids this in the prompt; `coach.curriculum_lock: true` enforces it — in the
release phase (success >= release_min_success for two reports) a proposal to lower a
curriculum lever is dropped, shown in the history, and the invariant releases instead.
