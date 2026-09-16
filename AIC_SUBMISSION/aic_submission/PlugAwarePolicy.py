from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from aic_submission.perception.plug_aware_inference import PlugAwarePoseInference
from aic_submission.perception.scene_pose_inference import ScenePoseInference


class PlugAwarePolicy(Policy):
    """CheatCode-style insertion using learned port and plug pose estimates."""

    PORT_Z_BY_PORT_TYPE = {
        "sfp": 0.133476,
        "sc": 0.014500,
    }
    PORT_XY_LIMITS_BY_PORT_TYPE = {
        "sfp": ((-0.54, -0.36), (0.15, 0.45)),
        "sc": ((-0.52, -0.33), (0.17, 0.34)),
    }

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._use_ground_truth_pose = (
            os.environ.get("AIC_PLUG_AWARE_USE_GROUND_TRUTH", "0") == "1"
        )
        self._debug_ground_truth = (
            os.environ.get("AIC_PLUG_AWARE_DEBUG_GROUND_TRUTH", "0") == "1"
        )

        self._use_scene_pose_model = (
            os.environ.get("AIC_PLUG_AWARE_SCENE_MODEL", "0") == "1"
        )
        run_name = (
            "scene_pose_100ep"
            if self._use_scene_pose_model
            else "plug_aware_pose_100ep"
        )
        default_checkpoint = Path(__file__).resolve().parents[1] / "runs" / run_name / "best_model.pt"
        checkpoint_path = Path(
            os.environ.get("AIC_PLUG_AWARE_CHECKPOINT", str(default_checkpoint))
        )
        self._predictor: PlugAwarePoseInference | None = None
        if checkpoint_path.exists() or not self._use_ground_truth_pose:
            device = os.environ.get("AIC_PLUG_AWARE_DEVICE", "auto")
            image_stride = int(os.environ.get("AIC_PLUG_AWARE_IMAGE_STRIDE", "4"))
            inference_cls = ScenePoseInference if self._use_scene_pose_model else PlugAwarePoseInference
            self._predictor = inference_cls(
                checkpoint_path,
                device=device,
                image_stride=image_stride,
            )
            self.get_logger().info(
                "Loaded plug-aware checkpoint "
                f"{self._predictor.checkpoint_path} on {self._predictor.device}; "
                f"epoch={self._predictor.checkpoint_epoch}, "
                f"val_loss={self._predictor.checkpoint_val_loss}"
            )
        else:
            self.get_logger().warn(
                "No plug-aware checkpoint loaded because ground-truth isolation "
                "mode is enabled."
            )

        if self._debug_ground_truth or self._use_ground_truth_pose:
            self.get_logger().warn(
                "Plug-aware ground-truth debug mode is enabled. Use this only "
                "for local diagnostics, not final evaluation."
            )

        self._filter_alpha = float(os.environ.get("AIC_PLUG_AWARE_FILTER_ALPHA", "0.35"))
        self._plug_filter_alpha = float(
            os.environ.get("AIC_PLUG_AWARE_PLUG_FILTER_ALPHA", "0.80")
        )
        self._clamp_port_z = os.environ.get("AIC_PLUG_AWARE_CLAMP_PORT_Z", "1") == "1"
        self._clamp_port_xy = os.environ.get("AIC_PLUG_AWARE_CLAMP_PORT_XY", "1") == "1"
        self._freeze_port_after = int(
            os.environ.get("AIC_PLUG_AWARE_FREEZE_PORT_AFTER", "0")
        )
        self._debug_log_every = max(
            1, int(os.environ.get("AIC_PLUG_AWARE_DEBUG_LOG_EVERY", "20"))
        )

        self._filtered_port_xyz: np.ndarray | None = None
        self._filtered_port_quat: np.ndarray | None = None
        self._filtered_plug_relative_xyz: np.ndarray | None = None
        self._filtered_plug_quat: np.ndarray | None = None
        self._prediction_count = 0
        self._debug_port_errors_mm: list[float] = []
        self._debug_plug_errors_mm: list[float] = []

        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05

    @staticmethod
    def _port_frame(task: Task) -> str:
        return f"task_board/{task.target_module_name}/{task.port_name}_link"

    @staticmethod
    def _plug_frame(task: Task) -> str:
        return f"{task.cable_name}/{task.plug_name}_link"

    @staticmethod
    def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
        quat = quat.astype(np.float64, copy=True)
        norm = np.linalg.norm(quat)
        if norm > 1.0e-8:
            quat /= norm
        if quat[3] < 0.0:
            quat *= -1.0
        return quat

    @staticmethod
    def _xyzw_to_wxyz(quat: np.ndarray) -> tuple[float, float, float, float]:
        quat = PlugAwarePolicy._normalize_quat_xyzw(quat)
        return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))

    @staticmethod
    def _wxyz_to_xyzw(quat: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
        quat_array = np.asarray(quat, dtype=np.float64)
        return PlugAwarePolicy._normalize_quat_xyzw(
            np.array(
                [quat_array[1], quat_array[2], quat_array[3], quat_array[0]],
                dtype=np.float64,
            )
        )

    @staticmethod
    def _blend_quat_xyzw(
        old_quat: np.ndarray | None,
        new_quat: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        new_quat = PlugAwarePolicy._normalize_quat_xyzw(new_quat)
        if old_quat is None:
            return new_quat
        old_quat = PlugAwarePolicy._normalize_quat_xyzw(old_quat)
        if np.dot(old_quat, new_quat) < 0.0:
            new_quat = -new_quat
        blended = (1.0 - alpha) * old_quat + alpha * new_quat
        return PlugAwarePolicy._normalize_quat_xyzw(blended)

    @staticmethod
    def _tcp_pose_from_observation(obs) -> Pose:
        return obs.controller_state.tcp_pose

    @staticmethod
    def _tcp_xyz_from_observation(obs) -> np.ndarray:
        pose = obs.controller_state.tcp_pose
        return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)

    def _wait_for_observation(
        self, get_observation: GetObservationCallback, timeout_sec: float = 10.0
    ):
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            obs = get_observation()
            if obs is not None and obs.left_image.height > 0:
                return obs
            self.get_logger().info("Waiting for observation...")
            time.sleep(0.1)
        return None

    def _lookup_pose_base(
        self,
        frame: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                frame,
                Time(),
            )
        except TransformException as ex:
            if self._prediction_count % self._debug_log_every == 0:
                self.get_logger().warn(f"Ground-truth lookup failed for {frame}: {ex}")
            return None

        tf = tf_stamped.transform
        xyz = np.array(
            [tf.translation.x, tf.translation.y, tf.translation.z],
            dtype=np.float32,
        )
        quat = np.array(
            [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w],
            dtype=np.float32,
        )
        return xyz, self._normalize_quat_xyzw(quat).astype(np.float32)

    def _predict_pose(self, obs, task: Task) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gt_port = None
        gt_plug = None
        if self._use_ground_truth_pose or self._debug_ground_truth:
            gt_port = self._lookup_pose_base(self._port_frame(task))
            gt_plug = self._lookup_pose_base(self._plug_frame(task))

        learned = None
        if not self._use_ground_truth_pose or self._debug_ground_truth:
            if self._predictor is not None:
                learned = self._predictor.predict(obs, task)

        if self._use_ground_truth_pose and gt_port is not None and gt_plug is not None:
            port_xyz, port_quat = gt_port
            plug_xyz, plug_quat = gt_plug
        elif learned is not None:
            port_xyz = learned.port_xyz
            port_quat = learned.port_quat_xyzw
            plug_xyz = learned.plug_xyz
            plug_quat = learned.plug_quat_xyzw
        else:
            raise RuntimeError("No plug-aware pose estimate is available.")

        if self._debug_ground_truth and learned is not None and gt_port is not None and gt_plug is not None:
            port_error_mm = float(np.linalg.norm(learned.port_xyz - gt_port[0]) * 1000.0)
            plug_error_mm = float(np.linalg.norm(learned.plug_xyz - gt_plug[0]) * 1000.0)
            self._debug_port_errors_mm.append(port_error_mm)
            self._debug_plug_errors_mm.append(plug_error_mm)
            if self._prediction_count % self._debug_log_every == 0:
                port_delta_mm = (learned.port_xyz - gt_port[0]) * 1000.0
                plug_delta_mm = (learned.plug_xyz - gt_plug[0]) * 1000.0
                self.get_logger().info(
                    "Plug-aware errors: "
                    f"port={port_error_mm:.1f}mm "
                    f"(dx={port_delta_mm[0]:.1f}, dy={port_delta_mm[1]:.1f}, dz={port_delta_mm[2]:.1f}), "
                    f"plug={plug_error_mm:.1f}mm "
                    f"(dx={plug_delta_mm[0]:.1f}, dy={plug_delta_mm[1]:.1f}, dz={plug_delta_mm[2]:.1f})"
                )

        alpha = float(np.clip(self._filter_alpha, 0.0, 1.0))
        plug_alpha = float(np.clip(self._plug_filter_alpha, 0.0, 1.0))
        tcp_xyz = self._tcp_xyz_from_observation(obs)
        plug_relative_xyz = plug_xyz - tcp_xyz

        freeze_port = (
            self._freeze_port_after > 0
            and self._prediction_count >= self._freeze_port_after
            and self._filtered_port_xyz is not None
        )
        if not freeze_port:
            if self._clamp_port_xy and task.port_type in self.PORT_XY_LIMITS_BY_PORT_TYPE:
                x_limits, y_limits = self.PORT_XY_LIMITS_BY_PORT_TYPE[task.port_type]
                port_xyz = port_xyz.copy()
                port_xyz[0] = np.clip(port_xyz[0], x_limits[0], x_limits[1])
                port_xyz[1] = np.clip(port_xyz[1], y_limits[0], y_limits[1])
            if self._clamp_port_z and task.port_type in self.PORT_Z_BY_PORT_TYPE:
                port_xyz = port_xyz.copy()
                port_xyz[2] = self.PORT_Z_BY_PORT_TYPE[task.port_type]

            if self._filtered_port_xyz is None:
                self._filtered_port_xyz = port_xyz.astype(np.float32)
            else:
                self._filtered_port_xyz = (
                    alpha * port_xyz + (1.0 - alpha) * self._filtered_port_xyz
                ).astype(np.float32)
            self._filtered_port_quat = self._blend_quat_xyzw(
                self._filtered_port_quat,
                port_quat,
                alpha,
            )

        if self._filtered_plug_relative_xyz is None:
            self._filtered_plug_relative_xyz = plug_relative_xyz.astype(np.float32)
        else:
            self._filtered_plug_relative_xyz = (
                plug_alpha * plug_relative_xyz
                + (1.0 - plug_alpha) * self._filtered_plug_relative_xyz
            ).astype(np.float32)

        self._filtered_plug_quat = self._blend_quat_xyzw(
            self._filtered_plug_quat,
            plug_quat,
            plug_alpha,
        )

        filtered_plug_xyz = tcp_xyz + self._filtered_plug_relative_xyz
        self._prediction_count += 1
        if self._prediction_count % 20 == 1:
            self.get_logger().info(
                "Plug-aware pose: "
                f"port=({self._filtered_port_xyz[0]:.4f}, "
                f"{self._filtered_port_xyz[1]:.4f}, "
                f"{self._filtered_port_xyz[2]:.4f}), "
                f"plug=({filtered_plug_xyz[0]:.4f}, "
                f"{filtered_plug_xyz[1]:.4f}, "
                f"{filtered_plug_xyz[2]:.4f})"
            )

        return (
            self._filtered_port_xyz.copy(),
            self._filtered_port_quat.copy(),
            filtered_plug_xyz.astype(np.float32),
            self._filtered_plug_quat.copy(),
        )

    def _target_pose_from_prediction(
        self,
        obs,
        port_xyz: np.ndarray,
        port_quat_xyzw: np.ndarray,
        plug_xyz: np.ndarray,
        plug_quat_xyzw: np.ndarray,
        z_offset: float,
        position_fraction: float = 1.0,
        slerp_fraction: float = 1.0,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        tcp_pose = self._tcp_pose_from_observation(obs)
        tcp_xyz = np.array(
            [tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z],
            dtype=np.float32,
        )
        tcp_quat_xyzw = np.array(
            [
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
            ],
            dtype=np.float32,
        )

        q_port = self._xyzw_to_wxyz(port_quat_xyzw)
        q_plug = self._xyzw_to_wxyz(plug_quat_xyzw)
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        q_tcp = self._xyzw_to_wxyz(tcp_quat_xyzw)
        q_tcp_target = quaternion_multiply(q_diff, q_tcp)
        q_tcp_slerp = quaternion_slerp(q_tcp, q_tcp_target, slerp_fraction)
        q_tcp_slerp_xyzw = self._wxyz_to_xyzw(q_tcp_slerp)

        plug_tip_gripper_offset = tcp_xyz - plug_xyz
        tip_x_error = float(port_xyz[0] - plug_xyz[0])
        tip_y_error = float(port_xyz[1] - plug_xyz[1])

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = float(
                np.clip(
                    self._tip_x_error_integrator + tip_x_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup,
                )
            )
            self._tip_y_error_integrator = float(
                np.clip(
                    self._tip_y_error_integrator + tip_y_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup,
                )
            )

        i_gain = 0.15
        target_xyz = np.array(
            [
                port_xyz[0] + i_gain * self._tip_x_error_integrator,
                port_xyz[1] + i_gain * self._tip_y_error_integrator,
                port_xyz[2] + z_offset - plug_tip_gripper_offset[2],
            ],
            dtype=np.float32,
        )
        blend_xyz = (
            position_fraction * target_xyz + (1.0 - position_fraction) * tcp_xyz
        )

        return Pose(
            position=Point(
                x=float(blend_xyz[0]),
                y=float(blend_xyz[1]),
                z=float(blend_xyz[2]),
            ),
            orientation=Quaternion(
                x=float(q_tcp_slerp_xyzw[0]),
                y=float(q_tcp_slerp_xyzw[1]),
                z=float(q_tcp_slerp_xyzw[2]),
                w=float(q_tcp_slerp_xyzw[3]),
            ),
        )

    def _set_predicted_pose_target(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        task: Task,
        z_offset: float,
        position_fraction: float = 1.0,
        slerp_fraction: float = 1.0,
        reset_xy_integrator: bool = False,
    ) -> bool:
        obs = get_observation()
        if obs is None:
            self.get_logger().warn("No observation available for plug-aware policy.")
            return False
        port_xyz, port_quat, plug_xyz, plug_quat = self._predict_pose(obs, task)
        pose = self._target_pose_from_prediction(
            obs,
            port_xyz,
            port_quat,
            plug_xyz,
            plug_quat,
            z_offset=z_offset,
            position_fraction=position_fraction,
            slerp_fraction=slerp_fraction,
            reset_xy_integrator=reset_xy_integrator,
        )
        self.set_pose_target(move_robot=move_robot, pose=pose)
        return True

    def _log_debug_summary(self) -> None:
        if not self._debug_port_errors_mm:
            return
        port_errors = np.asarray(self._debug_port_errors_mm, dtype=np.float32)
        plug_errors = np.asarray(self._debug_plug_errors_mm, dtype=np.float32)
        self.get_logger().info(
            "Plug-aware error summary: "
            f"port_mean={np.mean(port_errors):.1f}mm, "
            f"port_p90={np.percentile(port_errors, 90):.1f}mm, "
            f"plug_mean={np.mean(plug_errors):.1f}mm, "
            f"plug_p90={np.percentile(plug_errors, 90):.1f}mm"
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"PlugAwarePolicy.insert_cable() task: {task}")
        send_feedback("running plug-aware perception insertion")
        self._filtered_port_xyz = None
        self._filtered_port_quat = None
        self._filtered_plug_relative_xyz = None
        self._filtered_plug_quat = None
        self._prediction_count = 0
        self._debug_port_errors_mm = []
        self._debug_plug_errors_mm = []
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0

        if self._wait_for_observation(get_observation) is None:
            self.get_logger().error("No observation received before timeout.")
            return False

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

        while True:
            if z_offset < -0.015:
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

        self.get_logger().info("Waiting for connector to stabilize...")
        for _ in range(0, 20):
            self._set_predicted_pose_target(
                get_observation,
                move_robot,
                task,
                z_offset=z_offset,
            )
            self.sleep_for(0.25)

        self.get_logger().info("PlugAwarePolicy.insert_cable() exiting...")
        self._log_debug_summary()
        return True
