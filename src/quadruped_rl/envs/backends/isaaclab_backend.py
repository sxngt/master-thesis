"""Isaac Lab backend — PRIMARY TRAINING simulator (GPU-parallel).

Successor to legacy Isaac Gym Preview (deprecated, Python <= 3.8 only).
Requires an Isaac Lab installation (isaaclab + isaacsim packages); run
training with that environment's Python (see docs/setup.md).

Structure:
  IsaacLabEnv (our VectorEnv contract)  -- adapter, registered as "isaaclab"
    -> _QuadrupedEnv (isaaclab DirectRLEnv) -- scene, physics, reward, resets
       (mirrors isaaclab_tasks direct/anymal_c reference implementation)

Import discipline: isaaclab.envs/assets/... may only be imported AFTER the
AppLauncher has started Isaac Sim, so all heavy imports live inside
functions. Module import stays cheap so registry registration is safe.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from quadruped_rl.envs.base_env import VectorEnv
from quadruped_rl.registry import register_env_backend
from quadruped_rl.rewards.vectorized import VectorizedTraditionalReward

if importlib.util.find_spec("isaaclab") is None:  # keep backend unregistered
    raise ImportError("isaaclab is not installed")

_APP = None  # singleton SimulationApp (one per process)


def _launch_app(headless: bool, device: str) -> None:
    global _APP
    if _APP is None:
        from isaaclab.app import AppLauncher

        _APP = AppLauncher(headless=headless, device=device).app


# --------------------------------------------------------------------------- #
# Robot catalogue: maps configs/robot/<name>.yaml -> Isaac Lab assets.
# Only USD assets shipped with isaaclab_assets are supported for now;
# mini_cheetah needs a URDF conversion (docs/roadmap.md).
# --------------------------------------------------------------------------- #
_ROBOTS: dict[str, dict[str, str]] = {
    "a1": {
        "asset": "isaaclab_assets.robots.unitree:UNITREE_A1_CFG",
        "base": "trunk",
        "feet": ".*_foot",
    },
    "anymal_c": {
        "asset": "isaaclab_assets.robots.anymal:ANYMAL_C_CFG",
        "base": "base",
        "feet": ".*FOOT",
    },
}


def _load_robot_asset(name: str):
    if name not in _ROBOTS:
        raise NotImplementedError(
            f"Robot '{name}' has no Isaac Lab asset mapping. Supported: "
            f"{sorted(_ROBOTS)}. mini_cheetah requires URDF->USD conversion "
            "(see docs/roadmap.md)."
        )
    module_path, attr = _ROBOTS[name]["asset"].split(":")
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr), _ROBOTS[name]


# --------------------------------------------------------------------------- #
# Terrain mapping: configs/terrain/<name>.yaml (+ difficulty level)
#   -> isaaclab TerrainImporterCfg
# --------------------------------------------------------------------------- #
def _terrain_importer_cfg(terrain: dict[str, Any], level: str, friction: float):
    import isaaclab.sim as sim_utils
    import isaaclab.terrains as tg
    from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg

    name = terrain["name"]
    params = terrain.get("levels", {}).get(level, {})
    material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=friction,
        dynamic_friction=max(friction - 0.2, 0.05),
        restitution=0.0,
    )

    def importer(sub_terrain) -> TerrainImporterCfg:
        generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=2.0,
            num_rows=2,
            num_cols=2,
            sub_terrains={name: sub_terrain},
            use_cache=False,
        )
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=generator,
            collision_group=-1,
            physics_material=material,
            debug_vis=False,
        )

    if name in ("flat", "slippery"):
        # slippery: flat geometry, difficulty carried by the friction material
        return TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=material,
            debug_vis=False,
        )
    if name == "stairs":
        h = params["step_height_m"]
        return importer(
            tg.HfPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(h, h),
                step_width=params["step_depth_m"],
                platform_width=2.0,
            )
        )
    if name in ("slope", "rough_slope"):
        # rough_slope approximated by the sloped generator; surface noise TODO
        rad = float(np.radians(params["incline_deg"]))
        return importer(
            tg.HfPyramidSlopedTerrainCfg(proportion=1.0, slope_range=(rad, rad), platform_width=2.0)
        )
    if name == "gap":
        w = params["gap_width_m"]
        return importer(
            tg.MeshGapTerrainCfg(proportion=1.0, gap_width_range=(w, w), platform_width=2.0)
        )
    if name in ("gravel", "grass", "sand", "mud"):
        amp = (
            params.get("particle_size_m")
            or params.get("blade_height_m")
            or params.get("sink_depth_m", 0.02)
        )
        return importer(
            tg.HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=(-amp, amp),
                noise_step=max(amp / 4.0, 0.005),
                downsampled_scale=0.2,
            )
        )
    if name == "random_boxes":
        return importer(
            tg.MeshRandomGridTerrainCfg(
                proportion=1.0,
                grid_width=0.45,
                grid_height_range=(0.02, params["box_height_max_m"]),
                platform_width=2.0,
            )
        )
    raise NotImplementedError(
        f"Terrain '{name}' is not supported by the Isaac Lab backend yet "
        "(dynamic terrains moving_platform/seesaw need articulated stage "
        "assets — see docs/roadmap.md)."
    )


def _build_env_cfg(cfg: dict[str, Any], num_joints: int):
    """Translate our composed YAML config into a DirectRLEnvCfg instance."""
    import isaaclab.sim as sim_utils
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim import SimulationCfg

    sim, run = cfg["sim"], cfg["run"]
    robot_cfg, robot_meta = _load_robot_asset(cfg["robot"]["name"])
    terrain = cfg["terrain"]
    level = sim.get("terrain_level", "easy")
    friction = (
        terrain.get("levels", {}).get(level, {}).get("friction", terrain.get("friction", 0.8))
    )

    env_cfg = DirectRLEnvCfg(
        seed=run["seed"],
        decimation=sim["control_decimation"],
        episode_length_s=sim["episode_length_s"],
        action_space=num_joints,
        observation_space=3 + 3 + 3 + 3 + num_joints * 3,  # 48 for 12 joints
        state_space=0,
        sim=SimulationCfg(
            dt=sim["dt"],
            render_interval=sim["control_decimation"],
            device=run.get("device", "cuda"),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        scene=InteractiveSceneCfg(
            num_envs=sim["num_envs"], env_spacing=4.0, replicate_physics=True
        ),
    )
    # attached as plain attributes, consumed by _QuadrupedEnv
    env_cfg.robot = robot_cfg.replace(prim_path="/World/envs/env_.*/Robot")
    env_cfg.terrain = _terrain_importer_cfg(terrain, level, friction)
    env_cfg.contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=sim["dt"],
        track_air_time=True,
    )
    env_cfg.action_scale = sim.get("action_scale", 0.25)
    env_cfg.robot_meta = robot_meta
    return env_cfg


def _make_quadruped_env_class():
    """Define the DirectRLEnv subclass (deferred: needs a running app)."""
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv
    from isaaclab.sensors import ContactSensor
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

    class _QuadrupedEnv(DirectRLEnv):
        """Velocity-tracking quadruped locomotion on configurable terrain.

        Mirrors the isaaclab_tasks direct/anymal_c reference env, with the
        reward supplied by our config-driven VectorizedTraditionalReward and
        the extra state needed for the thesis KPI contract (Evaluator).
        """

        def __init__(
            self,
            cfg,
            reward_fn: VectorizedTraditionalReward,
            target_ms: float,
            course_length_m: float,
            **kwargs,
        ):
            self._reward_fn = reward_fn
            self._target_ms = target_ms
            self._course_length_m = course_length_m
            super().__init__(cfg, **kwargs)
            n, j = self.num_envs, self.cfg.action_space
            dev = self.device
            self._actions = torch.zeros(n, j, device=dev)
            self._previous_actions = torch.zeros(n, j, device=dev)
            self._start_pos = torch.zeros(n, 3, device=dev)
            self._fallen = torch.zeros(n, dtype=torch.bool, device=dev)
            meta = self.cfg.robot_meta
            self._base_id, _ = self._contact_sensor.find_bodies(meta["base"])
            self._feet_ids, _ = self._contact_sensor.find_bodies(meta["feet"])
            self._feet_body_ids, _ = self._robot.find_bodies(meta["feet"])

        # ----------------------------------------------------------- scene
        def _setup_scene(self):
            self._robot = Articulation(self.cfg.robot)
            self.scene.articulations["robot"] = self._robot
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
            self.scene.clone_environments(copy_from_source=False)
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
            light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light.func("/World/Light", light)

        # ---------------------------------------------------------- control
        def _pre_physics_step(self, actions: torch.Tensor):
            self._actions = actions.clone()
            self._processed_actions = (
                self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
            )

        def _apply_action(self):
            self._robot.set_joint_position_target(self._processed_actions)

        # ------------------------------------------------------------- obs
        def _get_observations(self) -> dict:
            self._previous_actions = self._actions.clone()
            data = self._robot.data
            commands = torch.zeros(self.num_envs, 3, device=self.device)
            commands[:, 0] = self._target_ms
            obs = torch.cat(
                [
                    data.root_lin_vel_b,
                    data.root_ang_vel_b,
                    data.projected_gravity_b,
                    commands,
                    data.joint_pos - data.default_joint_pos,
                    data.joint_vel,
                    self._actions,
                ],
                dim=-1,
            )
            return {"policy": obs}

        # ---------------------------------------------------------- reward
        def _reward_state(self) -> dict[str, torch.Tensor]:
            data = self._robot.data
            roll, pitch, yaw = euler_xyz_from_quat(data.root_quat_w)
            rpy = torch.stack([wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)], dim=-1)
            self._rpy = rpy
            # tangential foot velocity while in contact
            contact = self._contact_sensor.data.net_forces_w[:, self._feet_ids].norm(dim=-1) > 1.0
            feet_vel_xy = data.body_lin_vel_w[:, self._feet_body_ids, :2]
            slip = feet_vel_xy.norm(dim=-1) * contact.float()
            limits = data.soft_joint_pos_limits
            violation = torch.maximum(
                data.joint_pos - limits[..., 1], limits[..., 0] - data.joint_pos
            )
            return {
                "forward_velocity_ms": data.root_lin_vel_b[:, 0],
                "torques": data.applied_torque,
                "orientation_rpy": rpy,
                "foot_slip_velocity": slip,
                "joint_limit_violation": violation,
                "action_delta": self._actions - self._previous_actions,
                "fallen": self._fallen,
            }

        def _get_rewards(self) -> torch.Tensor:
            total, _ = self._reward_fn(self._reward_state())
            return total

        # ----------------------------------------------------------- dones
        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
            forces = self._contact_sensor.data.net_forces_w_history
            base_hit = torch.any(
                torch.max(torch.norm(forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
            )
            tipped = self._robot.data.projected_gravity_b[:, 2] > -0.5
            self._fallen = base_hit | tipped
            return self._fallen, time_out

        def _reset_idx(self, env_ids):
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = self._robot._ALL_INDICES
            self._robot.reset(env_ids)
            super()._reset_idx(env_ids)
            self._actions[env_ids] = 0.0
            self._previous_actions[env_ids] = 0.0
            joint_pos = self._robot.data.default_joint_pos[env_ids]
            joint_vel = self._robot.data.default_joint_vel[env_ids]
            root = self._robot.data.default_root_state[env_ids].clone()
            root[:, :3] += self._terrain.env_origins[env_ids]
            self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
            self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            self._start_pos[env_ids] = root[:, :3]
            self._fallen[env_ids] = False

        # ------------------------------------------- KPI info (Evaluator)
        def kpi_info(self) -> dict[str, torch.Tensor]:
            data = self._robot.data
            progress = (data.root_pos_w[:, :2] - self._start_pos[:, :2]).norm(dim=-1)
            feet_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids].norm(dim=-1)
            power = torch.sum(torch.abs(data.applied_torque * data.joint_vel), dim=-1)
            return {
                "positions": data.root_pos_w,
                "orientations_rpy": self._rpy,
                "torques": data.applied_torque,
                "joint_velocities": data.joint_vel,
                "contact_forces": feet_forces,
                "power_w": power,
                "falls": self._fallen.float(),
                "reached_goal": progress >= self._course_length_m,
                "goal_distance_m": torch.full_like(progress, self._course_length_m),
            }

    return _QuadrupedEnv


@register_env_backend("isaaclab")
class IsaacLabEnv(VectorEnv):
    """VectorEnv adapter over the DirectRLEnv-based quadruped task."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        sim = cfg["sim"]
        _launch_app(headless=sim.get("headless", True), device=cfg["run"].get("device", "cuda"))

        num_joints = cfg["robot"]["num_joints"]
        reward_cfg = cfg["reward"]
        if "components" not in reward_cfg:  # hybrid: use its traditional part
            reward_cfg = reward_cfg["traditional"]
            if "reward" in reward_cfg:
                reward_cfg = reward_cfg["reward"]
        reward_fn = VectorizedTraditionalReward(reward_cfg)
        target_ms = reward_cfg["components"].get("forward_velocity", {}).get("target_ms", 1.0)

        env_cfg = _build_env_cfg(cfg, num_joints)
        env_class = _make_quadruped_env_class()
        self._env = env_class(
            env_cfg,
            reward_fn=reward_fn,
            target_ms=target_ms,
            course_length_m=sim.get("course_length_m", 5.0),
        )
        self._obs_dim = env_cfg.observation_space
        self._act_dim = num_joints

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._act_dim

    @property
    def device(self) -> str:
        return str(self._env.device)

    def reset(self):
        obs_dict, _ = self._env.reset()
        return obs_dict["policy"]

    def step(self, actions):
        obs_dict, rewards, terminated, truncated, _ = self._env.step(actions)
        dones = terminated | truncated
        info = self._env.kpi_info()
        return obs_dict["policy"], rewards, dones, info

    def close(self) -> None:
        self._env.close()
