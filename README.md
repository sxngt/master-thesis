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

## Phase 1 Results — Which algorithm walks best?

Unitree A1, trained per algorithm with identical rewards and observations
(3 seeds; deterministic evaluation). Full analysis: [docs/findings.md](docs/findings.md),
raw table: `data/results/phase1_results_table.csv`.

| Algorithm | Flat v (m/s) | Flat succ. | Stairs v (m/s) | Stairs succ. | Seed std (flat) |
|---|---|---|---|---|---|
| **TRPO** | 1.021 ± 0.023 | 100% | **1.092 ± 0.012** | 100% | 0.023 |
| **PPO** | 1.030 ± 0.008 | 100% | 1.041 ± 0.006 | 100% | **0.008** |
| SAC | 0.806 ± 0.234 | 100% | 0.988 ± 0.025 | 100% | 0.234 |
| TD3 | 0.790 ± 0.315 | 100% | 0.956 ± 0.076 | 33% | 0.315 |
| DDPG (OU / param-noise) | 0.045 / 0.473 | 0% | – | – | – |

Stairs velocity differences are statistically significant
(ANOVA p=0.015, η²=0.71; Tukey: TRPO > TD3, d=2.5).

### Gait gallery (deterministic policies, best seed)

| | Flat | Stairs (easy) |
|---|---|---|
| PPO | ![PPO flat](docs/media/ppo_flat.gif) | ![PPO stairs](docs/media/ppo_stairs.gif) |
| TRPO | ![TRPO flat](docs/media/trpo_flat.gif) | ![TRPO stairs](docs/media/trpo_stairs.gif) |
| SAC | ![SAC flat](docs/media/sac_flat.gif) | ![SAC stairs](docs/media/sac_stairs.gif) |
| TD3 | ![TD3 flat](docs/media/td3_flat.gif) | ![TD3 stairs](docs/media/td3_stairs.gif) |

**Why DDPG fails** — exploration alone doesn't fix it (TD3's algorithmic changes do):

| DDPG + OU noise (0.045 m/s) | DDPG + parameter noise (0.473 m/s) |
|---|---|
| ![DDPG OU](docs/media/ddpg_ou_flat.gif) | ![DDPG param](docs/media/ddpg_parameter_space_flat.gif) |

High-quality MP4s of every clip are in [docs/media/](docs/media/).

