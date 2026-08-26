"""Simulation backend plugins.

Core research design: the SAME algorithms are validated across MULTIPLE
simulators. Each backend implements envs.base_env.BaseEnv and registers
itself via @register_env_backend("<name>"); the rest of the codebase
(algorithms, rewards, metrics, harness) never imports a simulator directly.

Backends with missing native deps are skipped silently — the mock backend
is always available.
"""

from quadruped_rl.envs.backends import mock  # noqa: F401  (always available)

for _mod in ("isaaclab_backend", "pybullet_backend", "gazebo"):
    try:
        __import__(f"quadruped_rl.envs.backends.{_mod}")
    except ImportError:
        pass
