"""Central registries for algorithms, environments, and reward components.

Registering a class makes it discoverable by name from configs, so the
matrix runner picks up new algorithms/terrains without code changes.
"""

from __future__ import annotations

from collections.abc import Callable

_ALGORITHMS: dict[str, type] = {}
_ENV_BACKENDS: dict[str, type] = {}
_REWARD_COMPONENTS: dict[str, Callable] = {}


def register_algorithm(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in _ALGORITHMS:
            raise ValueError(f"Algorithm '{name}' already registered")
        _ALGORITHMS[name] = cls
        return cls

    return deco


def get_algorithm(name: str) -> type:
    try:
        return _ALGORITHMS[name]
    except KeyError:
        raise KeyError(f"Unknown algorithm '{name}'. Registered: {sorted(_ALGORITHMS)}") from None


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHMS)


def register_env_backend(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _ENV_BACKENDS[name] = cls
        return cls

    return deco


def get_env_backend(name: str) -> type:
    try:
        return _ENV_BACKENDS[name]
    except KeyError:
        raise KeyError(
            f"Unknown env backend '{name}'. Registered: {sorted(_ENV_BACKENDS)}"
        ) from None


def register_reward_component(name: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        _REWARD_COMPONENTS[name] = fn
        return fn

    return deco


def get_reward_component(name: str) -> Callable:
    try:
        return _REWARD_COMPONENTS[name]
    except KeyError:
        raise KeyError(
            f"Unknown reward component '{name}'. Registered: {sorted(_REWARD_COMPONENTS)}"
        ) from None
