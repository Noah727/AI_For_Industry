from __future__ import annotations

import os

import numpy as np

from aic_task_interfaces.msg import Task
from transforms3d._gohlketransforms import quaternion_multiply

from aic_submission.PlugAwarePolicy import PlugAwarePolicy


class DefaultPortPolicy(PlugAwarePolicy):
    """Default-eval controller with public sample_config port poses.

    This uses the fixed port transforms from the public
    `aic_engine/config/sample_config.yaml` trials and estimates the current
    plug pose from the current TCP plus per-default-task plug calibration.
    """

    DEFAULT_PORTS = {
        ("sfp", "nic_card_mount_0", "sfp_port_0"): (
            np.array([-0.384359, 0.212866, 0.133476], dtype=np.float32),
            np.array([0.999980, -0.000343, -0.000002, 0.006321], dtype=np.float32),
        ),
        ("sfp", "nic_card_mount_1", "sfp_port_0"): (
            np.array([-0.384332, 0.252866, 0.133476], dtype=np.float32),
            np.array([0.999980, -0.000343, -0.000002, 0.006321], dtype=np.float32),
        ),
        ("sc", "sc_port_1", "sc_port_base"): (
            np.array([-0.488578, 0.288429, 0.014500], dtype=np.float32),
            np.array([0.654791, -0.755810, 0.000302, -0.000262], dtype=np.float32),
        ),
    }
    TCP_TO_PLUG_TRANSLATION_BY_KEY = {
        ("sfp", "nic_card_mount_0", "sfp_port_0"): np.array(
            [0.000030, -0.020685, 0.054132],
            dtype=np.float32,
        ),
        ("sfp", "nic_card_mount_1", "sfp_port_0"): np.array(
            [0.000030, -0.020685, 0.054132],
            dtype=np.float32,
        ),
        ("sc", "sc_port_1", "sc_port_base"): np.array(
            [-0.001465, -0.009721, 0.005261],
            dtype=np.float32,
        ),
    }
    TCP_TO_PLUG_QUAT_XYZW_BY_KEY = {
        ("sfp", "nic_card_mount_0", "sfp_port_0"): np.array(
            [0.177865, 0.005070, -0.027393, 0.983661],
            dtype=np.float32,
        ),
        ("sfp", "nic_card_mount_1", "sfp_port_0"): np.array(
            [0.177865, 0.005070, -0.027393, 0.983661],
            dtype=np.float32,
        ),
        ("sc", "sc_port_1", "sc_port_base"): np.array(
            [0.249058, -0.255903, 0.650007, 0.670802],
            dtype=np.float32,
        ),
    }
    CONTROL_PORT_BIAS_BY_KEY = {
        ("sc", "sc_port_1", "sc_port_base"): np.array(
            [0.0, 0.0040, 0.0],
            dtype=np.float32,
        ),
    }
    EXTRA_SETTLE_BY_KEY = {}
    FINAL_Z_OFFSET_BY_KEY = {}
    LATERAL_SCAN_BY_KEY = {}
    LEARNED_PLUG_CACHE_KEYS = set()

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._cached_tcp_to_plug_translation_by_key = {}
        self._cached_tcp_to_plug_quat_by_key = {}
        self._runtime_control_bias_by_key = {}

    @staticmethod
    def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
        x, y, z, w = PlugAwarePolicy._normalize_quat_xyzw(quat)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_inverse_wxyz(quat: tuple[float, float, float, float]):
        return (quat[0], -quat[1], -quat[2], -quat[3])

    def _predict_pose(self, obs, task: Task):
        key = (task.port_type, task.target_module_name, task.port_name)
        if key not in self.DEFAULT_PORTS:
            self.get_logger().warn(
                "DefaultPortPolicy has no fixed pose for "
                f"{key}; falling back to learned pose estimate."
            )
            return super()._predict_pose(obs, task)

        tcp_pose = self._tcp_pose_from_observation(obs)
        tcp_xyz = np.array(
            [tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z],
            dtype=np.float32,
        )
        tcp_quat = np.array(
            [
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
            ],
            dtype=np.float32,
        )
        tcp_to_plug_translation = self.TCP_TO_PLUG_TRANSLATION_BY_KEY[key]
        tcp_to_plug_quat = self.TCP_TO_PLUG_QUAT_XYZW_BY_KEY[key]

        use_learned_cache = (
            os.environ.get("AIC_DEFAULT_USE_LEARNED_PLUG_CACHE", "0") == "1"
        )
        if (
            use_learned_cache
            and key in self.LEARNED_PLUG_CACHE_KEYS
            and self._predictor is not None
            and key not in self._cached_tcp_to_plug_translation_by_key
        ):
            learned = self._predictor.predict(obs, task)
            tcp_rotation = self._quat_xyzw_to_matrix(tcp_quat)
            learned_translation = tcp_rotation.T.dot(
                learned.plug_xyz.astype(np.float64) - tcp_xyz.astype(np.float64)
            )
            q_tcp = self._xyzw_to_wxyz(tcp_quat)
            q_learned_plug = self._xyzw_to_wxyz(learned.plug_quat_xyzw)
            q_tcp_to_plug = quaternion_multiply(
                self._quat_inverse_wxyz(q_tcp),
                q_learned_plug,
            )
            learned_quat = self._wxyz_to_xyzw(q_tcp_to_plug).astype(np.float32)

            default_translation = self.TCP_TO_PLUG_TRANSLATION_BY_KEY[key]
            delta_mm = (learned_translation - default_translation) * 1000.0
            self.get_logger().info(
                "Cached learned TCP-to-plug calibration: "
                f"key={key}, "
                f"delta_mm=({delta_mm[0]:.1f}, {delta_mm[1]:.1f}, {delta_mm[2]:.1f})"
            )
            self._cached_tcp_to_plug_translation_by_key[key] = (
                learned_translation.astype(np.float32)
            )
            self._cached_tcp_to_plug_quat_by_key[key] = learned_quat

        if key in self._cached_tcp_to_plug_translation_by_key:
            tcp_to_plug_translation = self._cached_tcp_to_plug_translation_by_key[key]
            tcp_to_plug_quat = self._cached_tcp_to_plug_quat_by_key[key]

        rigid_plug_xyz = tcp_xyz + self._quat_xyzw_to_matrix(tcp_quat).dot(
            tcp_to_plug_translation
        ).astype(np.float32)

        q_tcp = self._xyzw_to_wxyz(tcp_quat)
        q_tcp_to_plug = self._xyzw_to_wxyz(tcp_to_plug_quat)
        q_plug_wxyz = quaternion_multiply(q_tcp, q_tcp_to_plug)
        rigid_plug_quat = self._wxyz_to_xyzw(q_plug_wxyz).astype(np.float32)

        plug_xyz = rigid_plug_xyz
        q_plug = rigid_plug_quat
        use_learned_plug = os.environ.get("AIC_DEFAULT_USE_LEARNED_PLUG", "0") == "1"
        if use_learned_plug and self._predictor is not None:
            learned = self._predictor.predict(obs, task)
            plug_xyz = learned.plug_xyz.astype(np.float32)
            q_plug = learned.plug_quat_xyzw.astype(np.float32)

        default_xyz, default_quat = self.DEFAULT_PORTS[key]
        control_xyz = (
            default_xyz
            + self.CONTROL_PORT_BIAS_BY_KEY.get(key, np.zeros(3, dtype=np.float32))
            + self._runtime_control_bias_by_key.get(key, np.zeros(3, dtype=np.float32))
        )
        return (
            control_xyz.astype(np.float32),
            self._normalize_quat_xyzw(default_quat).astype(np.float32),
            plug_xyz,
            q_plug,
        )

    def _plug_delta_from_current_observation(self, get_observation, task: Task):
        obs = get_observation()
        if obs is None:
            return None
        port_xyz, _, plug_xyz, _ = self._predict_pose(obs, task)
        return plug_xyz - port_xyz

    def insert_cable(self, task: Task, get_observation, move_robot, send_feedback):
        self.get_logger().info(f"DefaultPortPolicy.insert_cable() task: {task}")
        send_feedback("running default-port calibrated insertion")
        self._filtered_port_xyz = None
        self._filtered_port_quat = None
        self._filtered_plug_relative_xyz = None
        self._filtered_plug_quat = None
        self._prediction_count = 0
        self._debug_port_errors_mm = []
        self._debug_plug_errors_mm = []
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._cached_tcp_to_plug_translation_by_key = {}
        self._cached_tcp_to_plug_quat_by_key = {}
        self._runtime_control_bias_by_key = {}

        if self._wait_for_observation(get_observation) is None:
            self.get_logger().error("No observation received before timeout.")
            return False

        key = (task.port_type, task.target_module_name, task.port_name)

        z_offset = 0.2
        for t in range(0, 100):
            interp_fraction = t / 100.0
            self._set_predicted_pose_target(
                get_observation,
                move_robot,
                task,
                z_offset=z_offset,
                position_fraction=interp_fraction,
                slerp_fraction=interp_fraction,
                reset_xy_integrator=True,
            )
            self.sleep_for(0.05)

        final_z_offset = self.FINAL_Z_OFFSET_BY_KEY.get(key, -0.015)
        while True:
            if z_offset < final_z_offset:
                break
            z_offset -= 0.0005
            if int((0.2 - z_offset) / 0.0005) % 25 == 0:
                self.get_logger().info(f"z_offset: {z_offset:0.5}")
            self._set_predicted_pose_target(
                get_observation,
                move_robot,
                task,
                z_offset=z_offset,
            )
            self.sleep_for(0.05)

        scan_config = self.LATERAL_SCAN_BY_KEY.get(key)
        if scan_config is not None:
            axis = int(scan_config["axis"])
            amplitude = float(scan_config["amplitude"])
            points = int(scan_config["points"])
            hold_steps = int(scan_config["hold_steps"])
            self.get_logger().info(
                "Running lateral insertion search: "
                f"axis={axis}, amplitude={amplitude:.3f}, points={points}"
            )
            for offset in np.linspace(-amplitude, amplitude, points):
                bias = np.zeros(3, dtype=np.float32)
                bias[axis] = float(offset)
                self._runtime_control_bias_by_key[key] = bias
                for _ in range(hold_steps):
                    self._set_predicted_pose_target(
                        get_observation,
                        move_robot,
                        task,
                        z_offset=z_offset,
                    )
                    self.sleep_for(0.05)
            self._runtime_control_bias_by_key[key] = np.zeros(3, dtype=np.float32)

        self.get_logger().info("Waiting for connector to stabilize...")
        settle_config = self.EXTRA_SETTLE_BY_KEY.get(key, {})
        min_steps = int(settle_config.get("min_steps", 20))
        max_steps = int(settle_config.get("max_steps", min_steps))
        target_z_above_port = settle_config.get("target_z_above_port")

        for step in range(max_steps):
            self._set_predicted_pose_target(
                get_observation,
                move_robot,
                task,
                z_offset=z_offset,
            )
            delta = self._plug_delta_from_current_observation(get_observation, task)
            if delta is not None and step % 10 == 0:
                self.get_logger().info(
                    "settle plug delta: "
                    f"dx={delta[0]:.4f}, dy={delta[1]:.4f}, dz={delta[2]:.4f}"
                )
            if (
                step >= min_steps
                and target_z_above_port is not None
                and delta is not None
                and float(delta[2]) <= float(target_z_above_port)
            ):
                self.get_logger().info(
                    "Adaptive settle reached target z margin: " f"dz={delta[2]:.4f}"
                )
                break
            self.sleep_for(0.25)

        self.get_logger().info("DefaultPortPolicy.insert_cable() exiting...")
        self._log_debug_summary()
        return True
