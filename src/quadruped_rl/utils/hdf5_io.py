"""HDF5 trajectory storage (thesis 1.2.3 data-management spec).

Layout per file:
    /meta            attrs: run_id, robot, terrain, algorithm, seed
    /episodes/<i>/   datasets: joint_pos(1000Hz), joint_vel, torques, imu,
                     commands(100Hz), contacts, positions, orientations_rpy
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np


def save_episode(
    path: str, episode_idx: int, arrays: dict[str, np.ndarray], meta: dict[str, Any] | None = None
) -> None:
    with h5py.File(path, "a") as f:
        if meta:
            for k, v in meta.items():
                f.attrs[k] = v
        grp = f.require_group(f"episodes/{episode_idx}")
        for name, arr in arrays.items():
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=arr, compression="gzip")


def load_episode(path: str, episode_idx: int) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        grp = f[f"episodes/{episode_idx}"]
        return {name: np.asarray(ds) for name, ds in grp.items()}
