# Quadruped Rough-Terrain RL: Algorithm Comparison & LLM-Integrated Feedback

Master's thesis research codebase.

**Goals**
1. Systematic benchmark of RL algorithms (PPO, TRPO, A3C, SAC, TD3, DDPG) for
   quadruped rough-terrain locomotion across 3 robots × 12 terrains × 10 seeds.
2. LLM-integrated feedback RL: translating natural-language human feedback into
   reward signals via LLMs, combined with traditional rewards.

**Quick start**
```bash
make install          # uv sync (creates .venv, installs from uv.lock)
make test             # pure-logic unit tests
make smoke            # 1-minute end-to-end sanity run
```

See `CLAUDE.md` for conventions and `docs/` for protocols.
