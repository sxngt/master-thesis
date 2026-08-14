"""Quadruped rough-terrain RL: algorithm comparison & LLM-integrated feedback.

Importing this package registers all algorithm and env-backend plugins
(side effects of the subpackage imports below) so registry lookups work
from any entry point.
"""

__version__ = "0.1.0"

from quadruped_rl import algorithms, envs  # noqa: F401, E402  (plugin registration)
