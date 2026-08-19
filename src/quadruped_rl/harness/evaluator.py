"""Evaluation harness: rolls out a policy and computes the thesis KPI set.

All reported numbers in the thesis flow through this class so every
algorithm/robot/terrain combination is measured identically.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quadruped_rl.metrics.efficiency import cost_of_transport, torque_efficiency
from quadruped_rl.metrics.locomotion import (
    completion_time_s,
    mean_forward_velocity,
    path_efficiency,
    success_rate,
)
from quadruped_rl.metrics.stability import (
    attitude_stability,
    contact_force_variance,
    fall_frequency,
    recovery_time_s,
)


class Evaluator:
    def __init__(self, cfg: dict[str, Any], env):
        self.cfg = cfg
        self.env = env
        self.num_episodes = cfg["run"]["eval_episodes"]

    def run(self, algorithm) -> dict[str, float]:
        episodes = [self._rollout(algorithm) for _ in range(self.num_episodes)]
        return self.aggregate(episodes)

    def _rollout(self, algorithm) -> dict[str, Any]:
        obs = self.env.reset()
        traj: dict[str, list] = {
            k: []
            for k in (
                "positions",
                "orientations_rpy",
                "torques",
                "joint_velocities",
                "contact_forces",
                "falls",
                "power_w",
            )
        }
        done, steps = False, 0
        while not done:
            action = algorithm.act(obs, deterministic=True)
            obs, _, done, info = self.env.step(action)
            for key in traj:
                if key in info:
                    traj[key].append(info[key])
            steps += 1
        traj_np = {k: np.asarray(v) for k, v in traj.items() if v}
        traj_np["steps"] = steps
        traj_np["dt"] = self.env.control_dt
        traj_np["reached_goal"] = bool(info.get("reached_goal", False))
        traj_np["goal_distance_m"] = float(info.get("goal_distance_m", 0.0))
        return traj_np

    def aggregate(self, episodes: list[dict[str, Any]]) -> dict[str, float]:
        cfg_robot = self.cfg["robot"]
        out: dict[str, float] = {}
        out["success_rate"] = success_rate([e["reached_goal"] for e in episodes])

        vels, cots, path_effs, stabs = [], [], [], []
        for e in episodes:
            dt = e["dt"]
            if "positions" in e and len(e["positions"]) > 1:
                vels.append(mean_forward_velocity(e["positions"], dt))
                path_effs.append(path_efficiency(e["positions"], e["goal_distance_m"]))
            if "power_w" in e and "positions" in e:
                dist = np.linalg.norm(e["positions"][-1] - e["positions"][0])
                energy = float(np.sum(e["power_w"]) * dt)
                cots.append(cost_of_transport(energy, cfg_robot["mass_kg"], dist))
            if "orientations_rpy" in e:
                stabs.append(attitude_stability(e["orientations_rpy"]))

        for name, values in [
            ("mean_forward_velocity_ms", vels),
            ("cost_of_transport", cots),
            ("path_efficiency", path_effs),
            ("attitude_stability", stabs),
        ]:
            if values:
                out[name] = float(np.mean(values))

        completed = [completion_time_s(e["steps"], e["dt"]) for e in episodes if e["reached_goal"]]
        if completed:
            out["completion_time_s"] = float(np.mean(completed))

        falls = [
            fall_frequency(e.get("falls", np.zeros(1)), e["steps"] * e["dt"]) for e in episodes
        ]
        out["fall_frequency_per_min"] = float(np.mean(falls))

        extra = []
        for e in episodes:
            if "contact_forces" in e:
                extra.append(contact_force_variance(e["contact_forces"]))
        if extra:
            out["contact_force_variance"] = float(np.mean(extra))

        torq = []
        for e in episodes:
            if "torques" in e and "positions" in e and len(e["positions"]) > 1:
                dist = float(np.linalg.norm(e["positions"][-1] - e["positions"][0]))
                torq.append(torque_efficiency(e["torques"], dist))
        if torq:
            out["torque_efficiency"] = float(np.mean(torq))

        # recovery_time_s requires perturbation events; computed in robustness evals
        _ = recovery_time_s
        return out
