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

from aic_submission.perception.inference import RelativePortPoseInference
from aic_submission.perception.keypoint_inference import KeypointHeatmapInference
from aic_submission.perception.plug_aware_inference import PlugAwarePoseInference
from aic_submission.perception.target_pose_inference import TargetTcpPoseInference


class PerceptionGuidedPolicy(Policy):
    """Hybrid policy that replaces CheatCode's port TF with learned perception."""

    TCP_TO_PLUG_TRANSLATION_BY_PLUG_TYPE = {
        "sfp": np.array([0.000030, -0.020685, 0.054132], dtype=np.float32),
        "sc": np.array([-0.001465, -0.009721, 0.005261], dtype=np.float32),
    }
    TCP_TO_PLUG_QUAT_XYZW_BY_PLUG_TYPE = {
        "sfp": np.array([0.177865, 0.005070, -0.027393, 0.983661], dtype=np.float32),
        "sc": np.array([0.249058, -0.255903, 0.650007, 0.670802], dtype=np.float32),
    }
    PORT_QUAT_XYZW_BY_PORT_TYPE = {
        "sfp": np.array([0.999088, -0.012591, -0.000080, 0.006316], dtype=np.float32),
        "sc": np.array([-0.666766, 0.743857, -0.000298, 0.000267], dtype=np.float32),
    }
    PORT_Z_BY_PORT_TYPE = {
        "sfp": 0.133476,
        "sc": 0.014477,
    }

    def __init__(self, parent_node):
        super().__init__(parent_node)
        default_checkpoint = (
            Path(__file__).resolve().parents[1]
            / "runs"
            / "relative_port_pose_100ep"
            / "best_model.pt"
        )
        checkpoint_path = Path(
            os.environ.get("AIC_PERCEPTION_CHECKPOINT", str(default_checkpoint))
        )
        device = os.environ.get("AIC_PERCEPTION_DEVICE", "auto")
        image_stride = int(os.environ.get("AIC_PERCEPTION_IMAGE_STRIDE", "4"))
        self._predictor = RelativePortPoseInference(
            checkpoint_path, device=device, image_stride=image_stride
        )
        self._specialist_predictors: dict[str, RelativePortPoseInference] = {}
        for key, env_name in (
            ("sfp_port_0", "AIC_SFP_PORT0_PERCEPTION_CHECKPOINT"),
            ("sfp_port_1", "AIC_SFP_PORT1_PERCEPTION_CHECKPOINT"),
            ("sfp", "AIC_SFP_PERCEPTION_CHECKPOINT"),
            ("sc", "AIC_SC_PERCEPTION_CHECKPOINT"),
        ):
            specialist_path = os.environ.get(env_name, "").strip()
            if specialist_path:
                self._specialist_predictors[key] = RelativePortPoseInference(
                    Path(specialist_path),
                    device=device,
                    image_stride=image_stride,
                )
        self._orientation_predictor = None
        if os.environ.get("AIC_PERCEPTION_ORIENTATION_SOURCE", "fixed") == "target_pose":
            default_orientation_checkpoint = (
                Path(__file__).resolve().parents[1]
                / "runs"
                / "target_pose_100ep"
                / "best_model.pt"
            )
            orientation_checkpoint_path = Path(
                os.environ.get(
                    "AIC_TARGET_POSE_CHECKPOINT",
                    str(default_orientation_checkpoint),
                )
            )
            self._orientation_predictor = TargetTcpPoseInference(
                orientation_checkpoint_path,
                device=device,
                image_stride=image_stride,
            )
        self._plug_predictor = None
        if os.environ.get("AIC_PERCEPTION_USE_LEARNED_PLUG", "0") == "1":
            default_plug_checkpoint = (
                Path(__file__).resolve().parents[1]
                / "runs"
                / "plug_aware_pose_100ep"
                / "best_model.pt"
            )
            plug_checkpoint_path = Path(
                os.environ.get(
                    "AIC_PLUG_AWARE_CHECKPOINT",
                    str(default_plug_checkpoint),
                )
            )
            self._plug_predictor = PlugAwarePoseInference(
                plug_checkpoint_path,
                device=device,
                image_stride=image_stride,
            )
        self._keypoint_predictor = None
        if os.environ.get("AIC_PERCEPTION_USE_KEYPOINT", "0") == "1":
            default_keypoint_checkpoint = (
                Path(__file__).resolve().parents[1]
                / "runs"
                / "keypoint_center_100ep"
                / "best_model.pt"
            )
            keypoint_checkpoint_path = Path(
                os.environ.get(
                    "AIC_PERCEPTION_KEYPOINT_CHECKPOINT",
                    str(default_keypoint_checkpoint),
                )
            )
            self._keypoint_predictor = KeypointHeatmapInference(
                keypoint_checkpoint_path,
                device=os.environ.get("AIC_PERCEPTION_KEYPOINT_DEVICE", device),
                image_key=os.environ.get(
                    "AIC_PERCEPTION_KEYPOINT_IMAGE_KEY", "center_image"
                ),
                image_stride=int(
                    os.environ.get("AIC_PERCEPTION_KEYPOINT_IMAGE_STRIDE", "4")
                ),
            )
        self._filter_alpha = float(os.environ.get("AIC_PORT_FILTER_ALPHA", "0.35"))
        self._plug_filter_alpha = float(
            os.environ.get("AIC_PERCEPTION_PLUG_FILTER_ALPHA", "0.80")
        )
        self._learned_plug_mode = os.environ.get(
            "AIC_PERCEPTION_LEARNED_PLUG_MODE", "xyz"
        ).strip().lower()
        self._learned_plug_after_z = float(
            os.environ.get("AIC_PERCEPTION_LEARNED_PLUG_AFTER_Z", "0.03")
        )
        self._learned_plug_gate_m = float(
            os.environ.get("AIC_PERCEPTION_LEARNED_PLUG_GATE_M", "0.025")
        )
        self._debug_ground_truth = os.environ.get("AIC_DEBUG_GROUND_TRUTH", "0") == "1"
        self._use_ground_truth_port = (
            os.environ.get("AIC_USE_GROUND_TRUTH_PORT", "0") == "1"
        )
        self._debug_log_every = max(
            1, int(os.environ.get("AIC_DEBUG_LOG_EVERY", "20"))
        )
        self._freeze_port_after = int(
            os.environ.get("AIC_PERCEPTION_FREEZE_PORT_AFTER", "0")
        )
        self._final_z_offset_by_port_type = {
            "sfp": float(os.environ.get("AIC_SFP_FINAL_Z_OFFSET", "-0.015")),
            "sc": float(os.environ.get("AIC_SC_FINAL_Z_OFFSET", "-0.015")),
        }
        self._insert_step = float(os.environ.get("AIC_PERCEPTION_INSERT_STEP", "0.0005"))
        self._lateral_search_amplitude_by_port_type = {
            "sfp": float(os.environ.get("AIC_SFP_LATERAL_SEARCH_AMPLITUDE", "0.0")),
            "sc": float(os.environ.get("AIC_SC_LATERAL_SEARCH_AMPLITUDE", "0.0")),
        }
        self._lateral_search_hold_steps = int(
            os.environ.get("AIC_LATERAL_SEARCH_HOLD_STEPS", "3")
        )
        self._lateral_search_retract_steps = int(
            os.environ.get("AIC_LATERAL_SEARCH_RETRACT_STEPS", "6")
        )
        self._lateral_search_insert_step = float(
            os.environ.get("AIC_LATERAL_SEARCH_INSERT_STEP", "0.004")
        )
        self._lateral_search_entry_z_by_port_type = {
            "sfp": float(os.environ.get("AIC_SFP_LATERAL_SEARCH_ENTRY_Z_OFFSET", "nan")),
            "sc": float(os.environ.get("AIC_SC_LATERAL_SEARCH_ENTRY_Z_OFFSET", "nan")),
        }
        self._keep_lateral_search_bias = (
            os.environ.get("AIC_LATERAL_SEARCH_KEEP_LAST_BIAS", "0") == "1"
        )
        self._lateral_search_patterns = {
            "sfp": self._parse_lateral_search_pattern(
                os.environ.get("AIC_SFP_LATERAL_SEARCH_PATTERN", "")
            ),
            "sc": self._parse_lateral_search_pattern(
                os.environ.get("AIC_SC_LATERAL_SEARCH_PATTERN", "")
            ),
        }
        self._port_bias_by_target = self._parse_named_bias_map(
            os.environ.get("AIC_PORT_BIAS_BY_TARGET", "")
        )
        self._port_bias_after_z = float(os.environ.get("AIC_PORT_BIAS_AFTER_Z", "inf"))
        self._keypoint_min_confidence = float(
            os.environ.get("AIC_PERCEPTION_KEYPOINT_MIN_CONFIDENCE", "0.35")
        )
        self._keypoint_gain = float(
            os.environ.get("AIC_PERCEPTION_KEYPOINT_CORRECTION_GAIN", "1.0")
        )
        self._keypoint_max_correction_m = float(
            os.environ.get("AIC_PERCEPTION_KEYPOINT_MAX_CORRECTION_M", "0.050")
        )
        self._keypoint_filter_alpha = float(
            os.environ.get("AIC_PERCEPTION_KEYPOINT_FILTER_ALPHA", "0.45")
        )
        self._keypoint_enable_after_count = int(
            os.environ.get("AIC_PERCEPTION_KEYPOINT_ENABLE_AFTER_COUNT", "10")
        )
        self._keypoint_update_min_z_offset = float(
            os.environ.get("AIC_PERCEPTION_KEYPOINT_UPDATE_MIN_Z_OFFSET", "0.020")
        )
        self._keypoint_max_delta_px = float(
            os.environ.get("AIC_PERCEPTION_KEYPOINT_MAX_DELTA_PX", "420")
        )
        self._keypoint_max_raw_correction_m = float(
            os.environ.get("AIC_PERCEPTION_KEYPOINT_MAX_RAW_CORRECTION_M", "0.100")
        )
        self._filtered_port_xyz: np.ndarray | None = None
        self._frozen_port_xyz: np.ndarray | None = None
        self._frozen_target_quat_xyzw: np.ndarray | None = None
        self._filtered_plug_relative_xyz: np.ndarray | None = None
        self._filtered_visual_correction: np.ndarray | None = None
        self._runtime_port_bias = np.zeros(3, dtype=np.float32)
        self._prediction_count = 0
        self._debug_errors_mm: list[float] = []
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05

        self.get_logger().info(
            "Loaded perception checkpoint "
            f"{self._predictor.checkpoint_path} on {self._predictor.device}; "
            f"epoch={self._predictor.checkpoint_epoch}, "
            f"val_loss={self._predictor.checkpoint_val_loss}; "
            f"freeze_after={self._freeze_port_after}; "
            f"final_z={self._final_z_offset_by_port_type}; "
            f"lateral_search={self._lateral_search_amplitude_by_port_type}"
        )
        if self._port_bias_by_target:
            self.get_logger().info(
                "Using target-specific port bias map: "
                f"{self._port_bias_by_target}"
            )
        for key, predictor in self._specialist_predictors.items():
            self.get_logger().info(
                "Loaded specialist perception checkpoint "
                f"{key}: {predictor.checkpoint_path}; "
                f"epoch={predictor.checkpoint_epoch}, "
                f"val_loss={predictor.checkpoint_val_loss}"
            )
        if self._orientation_predictor is not None:
            self.get_logger().info(
                "Using target-pose model for TCP orientation: "
                f"{self._orientation_predictor.checkpoint_path}"
            )
        if self._plug_predictor is not None:
            self.get_logger().info(
                "Using plug-aware model for plug XYZ only: "
                f"{self._plug_predictor.checkpoint_path}; "
                f"plug_alpha={self._plug_filter_alpha}; "
                f"plug_mode={self._learned_plug_mode}; "
                f"after_z={self._learned_plug_after_z}; "
                f"gate_m={self._learned_plug_gate_m}"
            )
        if self._keypoint_predictor is not None:
            self.get_logger().info(
                "Using keypoint visual servo model: "
                f"{self._keypoint_predictor.checkpoint_path}; "
                f"val_px={self._keypoint_predictor.checkpoint_val_pixel_error}; "
                f"gain={self._keypoint_gain}; "
                f"max_correction={self._keypoint_max_correction_m:.3f}m"
            )
        if self._debug_ground_truth or self._use_ground_truth_port:
            self.get_logger().warn(
                "Ground-truth debug mode is enabled. Use this only for local "
                "diagnostics, not final evaluation."
            )

    @staticmethod
    def _port_frame(task: Task) -> str:
        return f"task_board/{task.target_module_name}/{task.port_name}_link"

    def _lookup_ground_truth_port_xyz(self, task: Task) -> np.ndarray | None:
        try:
            tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                self._port_frame(task),
                Time(),
            )
        except TransformException as ex:
            if self._prediction_count % self._debug_log_every == 0:
                self.get_logger().warn(f"Ground-truth port lookup failed: {ex}")
            return None

        translation = tf_stamped.transform.translation
        return np.array([translation.x, translation.y, translation.z], dtype=np.float32)

    def _predictor_for_task(self, task: Task) -> RelativePortPoseInference:
        if task.port_type == "sc":
            return self._specialist_predictors.get("sc", self._predictor)
        if task.port_type == "sfp":
            port_specific = self._specialist_predictors.get(task.port_name)
            if port_specific is not None:
                return port_specific
            return self._specialist_predictors.get("sfp", self._predictor)
        return self._predictor

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

    def _predict_port_xyz(self, obs, task: Task) -> np.ndarray:
        if self._frozen_port_xyz is not None:
            self._prediction_count += 1
            if self._prediction_count % self._debug_log_every == 1:
                self.get_logger().info(
                    "Using frozen port xyz: "
                    f"{self._frozen_port_xyz[0]:.4f}, "
                    f"{self._frozen_port_xyz[1]:.4f}, "
                    f"{self._frozen_port_xyz[2]:.4f}"
                )
            return self._frozen_port_xyz.copy()

        ground_truth_port_xyz = None
        if self._debug_ground_truth or self._use_ground_truth_port:
            ground_truth_port_xyz = self._lookup_ground_truth_port_xyz(task)

        learned_port_xyz = None
        if not self._use_ground_truth_port or self._debug_ground_truth:
            learned_port_xyz = self._predictor_for_task(task).predict_port_xyz(obs, task)

        if self._use_ground_truth_port and ground_truth_port_xyz is not None:
            raw_port_xyz = ground_truth_port_xyz
        elif learned_port_xyz is not None:
            raw_port_xyz = learned_port_xyz
        else:
            self.get_logger().warn("Falling back to filtered port estimate.")
            if self._filtered_port_xyz is None:
                raw_port_xyz = RelativePortPoseInference.tcp_xyz_from_observation(obs)
            else:
                raw_port_xyz = self._filtered_port_xyz

        if self._debug_ground_truth and ground_truth_port_xyz is not None:
            debug_port_xyz = (
                learned_port_xyz if learned_port_xyz is not None else raw_port_xyz
            )
            error_mm = float(
                np.linalg.norm(debug_port_xyz - ground_truth_port_xyz) * 1000.0
            )
            self._debug_errors_mm.append(error_mm)
            if self._prediction_count % self._debug_log_every == 0:
                delta_mm = (debug_port_xyz - ground_truth_port_xyz) * 1000.0
                self.get_logger().info(
                    "Learned-vs-GT port error: "
                    f"{error_mm:.1f} mm "
                    f"(dx={delta_mm[0]:.1f}, dy={delta_mm[1]:.1f}, dz={delta_mm[2]:.1f})"
                )

        if self._filtered_port_xyz is None:
            self._filtered_port_xyz = raw_port_xyz
        else:
            alpha = np.clip(self._filter_alpha, 0.0, 1.0)
            self._filtered_port_xyz = (
                alpha * raw_port_xyz + (1.0 - alpha) * self._filtered_port_xyz
            )
        if task.port_type in self.PORT_Z_BY_PORT_TYPE:
            self._filtered_port_xyz[2] = self.PORT_Z_BY_PORT_TYPE[task.port_type]
        self._prediction_count += 1
        if (
            self._freeze_port_after > 0
            and self._prediction_count >= self._freeze_port_after
            and self._frozen_port_xyz is None
        ):
            self._frozen_port_xyz = self._filtered_port_xyz.copy()
            self.get_logger().info(
                "Freezing port xyz after "
                f"{self._prediction_count} predictions: "
                f"{self._frozen_port_xyz[0]:.4f}, "
                f"{self._frozen_port_xyz[1]:.4f}, "
                f"{self._frozen_port_xyz[2]:.4f}"
            )
        if self._prediction_count % 20 == 1:
            self.get_logger().info(
                "Predicted port xyz: "
                f"{self._filtered_port_xyz[0]:.4f}, "
                f"{self._filtered_port_xyz[1]:.4f}, "
                f"{self._filtered_port_xyz[2]:.4f}"
            )
        return self._filtered_port_xyz.copy()

    def _predict_target_quat_xyzw(self, obs, task: Task) -> np.ndarray | None:
        if self._orientation_predictor is None:
            return None
        if self._frozen_target_quat_xyzw is not None:
            return self._frozen_target_quat_xyzw.copy()
        target_pose = self._orientation_predictor.predict_target_pose(obs, task)
        quat = np.array(
            [
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
                target_pose.orientation.w,
            ],
            dtype=np.float32,
        )
        quat = self._normalize_quat_xyzw(quat).astype(np.float32)
        if self._freeze_port_after > 0 and self._prediction_count >= self._freeze_port_after:
            self._frozen_target_quat_xyzw = quat.copy()
            self.get_logger().info(
                "Freezing target TCP orientation: "
                f"{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}"
            )
        return quat

    def _predict_plug_xyz(self, obs, task: Task) -> np.ndarray | None:
        if self._plug_predictor is None:
            return None
        learned = self._plug_predictor.predict(obs, task)
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
        plug_type = (
            task.plug_type
            if task.plug_type in self.TCP_TO_PLUG_TRANSLATION_BY_PLUG_TYPE
            else "sfp"
        )
        tcp_rotation = self._quat_xyzw_to_matrix(tcp_quat_xyzw)
        nominal_plug_xyz = tcp_xyz + tcp_rotation.dot(
            self.TCP_TO_PLUG_TRANSLATION_BY_PLUG_TYPE[plug_type]
        ).astype(np.float32)
        plug_delta = learned.plug_xyz.astype(np.float32) - nominal_plug_xyz
        if self._learned_plug_gate_m > 0.0 and np.linalg.norm(plug_delta) > self._learned_plug_gate_m:
            if self._prediction_count % self._debug_log_every == 0:
                self.get_logger().warn(
                    "Rejecting learned plug outlier: "
                    f"delta={np.linalg.norm(plug_delta) * 1000.0:.1f} mm"
                )
            return None
        plug_relative_xyz = learned.plug_xyz.astype(np.float32) - tcp_xyz
        if self._filtered_plug_relative_xyz is None:
            self._filtered_plug_relative_xyz = plug_relative_xyz
        else:
            alpha = float(np.clip(self._plug_filter_alpha, 0.0, 1.0))
            self._filtered_plug_relative_xyz = (
                alpha * plug_relative_xyz
                + (1.0 - alpha) * self._filtered_plug_relative_xyz
            ).astype(np.float32)
        return (tcp_xyz + self._filtered_plug_relative_xyz).astype(np.float32)

    def _log_debug_summary(self) -> None:
        if not self._debug_errors_mm:
            return
        errors = np.asarray(self._debug_errors_mm, dtype=np.float32)
        self.get_logger().info(
            "Learned-vs-GT port error summary: "
            f"count={errors.size}, mean={np.mean(errors):.1f} mm, "
            f"median={np.median(errors):.1f} mm, "
            f"p90={np.percentile(errors, 90):.1f} mm, "
            f"max={np.max(errors):.1f} mm"
        )

    @staticmethod
    def _tcp_pose_from_observation(obs) -> Pose:
        return obs.controller_state.tcp_pose

    @staticmethod
    def _parse_lateral_search_pattern(pattern: str) -> list[tuple[float, float, float]]:
        offsets = []
        for chunk in pattern.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [part.strip() for part in chunk.split(",")]
            if len(parts) not in (2, 3):
                continue
            try:
                dx = float(parts[0])
                dy = float(parts[1])
                dz = float(parts[2]) if len(parts) == 3 else 0.0
                offsets.append((dx, dy, dz))
            except ValueError:
                continue
        return offsets

    @staticmethod
    def _parse_named_bias_map(pattern: str) -> dict[str, np.ndarray]:
        biases: dict[str, np.ndarray] = {}
        for chunk in pattern.split(";"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            name, values = chunk.split(":", 1)
            parts = [part.strip() for part in values.split(",")]
            if len(parts) != 3:
                continue
            try:
                biases[name.strip()] = np.array(
                    [float(parts[0]), float(parts[1]), float(parts[2])],
                    dtype=np.float32,
                )
            except ValueError:
                continue
        return biases

    def _static_port_bias_for_task(self, task: Task) -> np.ndarray:
        """Return the most specific configured base-frame port correction."""
        candidate_keys = (
            f"{task.target_module_name}/{task.port_name}",
            f"{task.target_module_name}.{task.port_name}",
            task.port_name,
            task.target_module_name,
            task.port_type,
            task.plug_type,
        )
        for key in candidate_keys:
            bias = self._port_bias_by_target.get(key)
            if bias is not None:
                return bias
        return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _clamp_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm > max_norm > 0.0:
            return vector * (max_norm / norm)
        return vector

    def _has_lateral_search(self, port_type: str) -> bool:
        offsets = self._lateral_search_patterns.get(port_type, [])
        amplitude = self._lateral_search_amplitude_by_port_type.get(port_type, 0.0)
        return bool(offsets) or amplitude > 0.0

    def _apply_visual_correction(self, port_xyz: np.ndarray) -> np.ndarray:
        if self._filtered_visual_correction is None:
            return port_xyz
        corrected = port_xyz.copy()
        corrected[:2] = (
            corrected[:2]
            + self._keypoint_gain * self._filtered_visual_correction[:2]
        )
        return corrected.astype(np.float32)

    def _visual_corrected_port_xyz(
        self,
        obs,
        task: Task,
        port_xyz: np.ndarray,
        plug_xyz: np.ndarray,
        z_offset: float,
    ) -> np.ndarray:
        if self._keypoint_predictor is None:
            return port_xyz
        if self._prediction_count < self._keypoint_enable_after_count:
            return self._apply_visual_correction(port_xyz)
        if z_offset < self._keypoint_update_min_z_offset:
            return self._apply_visual_correction(port_xyz)

        prediction = self._keypoint_predictor.predict(obs, task)
        if prediction.min_confidence < self._keypoint_min_confidence:
            if self._prediction_count % self._debug_log_every == 0:
                self.get_logger().warn(
                    "Skipping keypoint correction: low confidence "
                    f"port={prediction.port_confidence:.3f}, "
                    f"plug={prediction.plug_confidence:.3f}"
                )
            return self._apply_visual_correction(port_xyz)

        delta_uv_norm = float(np.linalg.norm(prediction.delta_uv_full))
        if delta_uv_norm > self._keypoint_max_delta_px:
            if self._prediction_count % self._debug_log_every == 0:
                self.get_logger().warn(
                    "Skipping keypoint correction: implausible pixel delta "
                    f"{delta_uv_norm:.1f}px"
                )
            return self._apply_visual_correction(port_xyz)

        visual_delta_base = self._keypoint_predictor.pixel_delta_to_base_vector(
            obs=obs,
            image_key=prediction.image_key,
            delta_uv_full=prediction.delta_uv_full,
            reference_point_base=plug_xyz,
        )
        if visual_delta_base is None or not np.all(np.isfinite(visual_delta_base)):
            return self._apply_visual_correction(port_xyz)

        desired_correction = (plug_xyz + visual_delta_base) - port_xyz
        desired_correction = desired_correction.astype(np.float32)
        desired_correction[2] = 0.0
        raw_correction_norm = float(np.linalg.norm(desired_correction))
        if raw_correction_norm > self._keypoint_max_raw_correction_m:
            if self._prediction_count % self._debug_log_every == 0:
                self.get_logger().warn(
                    "Skipping keypoint correction: raw correction too large "
                    f"{raw_correction_norm * 1000.0:.1f}mm"
                )
            return self._apply_visual_correction(port_xyz)

        desired_correction = self._clamp_norm(
            desired_correction,
            self._keypoint_max_correction_m,
        ).astype(np.float32)

        alpha = float(np.clip(self._keypoint_filter_alpha, 0.0, 1.0))
        if self._filtered_visual_correction is None:
            self._filtered_visual_correction = desired_correction
        else:
            self._filtered_visual_correction = (
                alpha * desired_correction
                + (1.0 - alpha) * self._filtered_visual_correction
            ).astype(np.float32)

        corrected = self._apply_visual_correction(port_xyz)
        if self._prediction_count % self._debug_log_every == 0:
            self.get_logger().info(
                "Keypoint correction: "
                f"duv=({prediction.delta_uv_full[0]:.1f}, "
                f"{prediction.delta_uv_full[1]:.1f})px, "
                f"conf=({prediction.port_confidence:.2f}, "
                f"{prediction.plug_confidence:.2f}), "
                f"raw={raw_correction_norm * 1000.0:.1f}mm, "
                f"corr=({(corrected[0] - port_xyz[0]) * 1000.0:.1f}, "
                f"{(corrected[1] - port_xyz[1]) * 1000.0:.1f})mm"
            )
        return corrected.astype(np.float32)

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
        quat = PerceptionGuidedPolicy._normalize_quat_xyzw(quat)
        return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))

    @staticmethod
    def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
        return PerceptionGuidedPolicy._normalize_quat_xyzw(
            np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
        )

    @staticmethod
    def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
        x, y, z, w = PerceptionGuidedPolicy._normalize_quat_xyzw(quat)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _target_pose_from_prediction(
        self,
        obs,
        task: Task,
        predicted_port_xyz: np.ndarray,
        z_offset: float,
        position_fraction: float = 1.0,
        slerp_fraction: float = 1.0,
        reset_xy_integrator: bool = False,
        target_tcp_quat_xyzw: np.ndarray | None = None,
        learned_plug_xyz: np.ndarray | None = None,
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

        plug_type = (
            task.plug_type if task.plug_type in self.TCP_TO_PLUG_TRANSLATION_BY_PLUG_TYPE else "sfp"
        )
        port_type = task.port_type if task.port_type in self.PORT_QUAT_XYZW_BY_PORT_TYPE else plug_type

        tcp_rotation = self._quat_xyzw_to_matrix(tcp_quat_xyzw)
        tcp_to_plug_translation = self.TCP_TO_PLUG_TRANSLATION_BY_PLUG_TYPE[plug_type]
        rigid_plug_xyz = tcp_xyz + tcp_rotation.dot(tcp_to_plug_translation).astype(
            np.float32
        )
        if learned_plug_xyz is None:
            plug_xyz = rigid_plug_xyz
        elif self._learned_plug_mode == "xy":
            plug_xyz = np.array(
                [learned_plug_xyz[0], learned_plug_xyz[1], rigid_plug_xyz[2]],
                dtype=np.float32,
            )
        else:
            plug_xyz = learned_plug_xyz.astype(np.float32)
        plug_tip_gripper_offset = tcp_xyz - plug_xyz
        if z_offset <= self._port_bias_after_z:
            static_port_bias = self._static_port_bias_for_task(task)
        else:
            static_port_bias = np.zeros(3, dtype=np.float32)
        corrected_port_xyz = self._visual_corrected_port_xyz(
            obs,
            task,
            predicted_port_xyz,
            plug_xyz,
            z_offset,
        )
        control_port_xyz = (
            corrected_port_xyz + static_port_bias + self._runtime_port_bias
        )

        tip_x_error = float(control_port_xyz[0] - plug_xyz[0])
        tip_y_error = float(control_port_xyz[1] - plug_xyz[1])
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
                control_port_xyz[0] + i_gain * self._tip_x_error_integrator,
                control_port_xyz[1] + i_gain * self._tip_y_error_integrator,
                control_port_xyz[2] + z_offset - plug_tip_gripper_offset[2],
            ],
            dtype=np.float32,
        )
        blended_xyz = (
            position_fraction * target_xyz + (1.0 - position_fraction) * tcp_xyz
        )

        q_tcp = self._xyzw_to_wxyz(tcp_quat_xyzw)
        if target_tcp_quat_xyzw is None:
            q_port = self._xyzw_to_wxyz(self.PORT_QUAT_XYZW_BY_PORT_TYPE[port_type])
            q_tcp_to_plug = self._xyzw_to_wxyz(
                self.TCP_TO_PLUG_QUAT_XYZW_BY_PLUG_TYPE[plug_type]
            )
            q_plug = quaternion_multiply(q_tcp, q_tcp_to_plug)
            q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
            q_diff = quaternion_multiply(q_port, q_plug_inv)
            q_target = quaternion_multiply(q_diff, q_tcp)
        else:
            q_target = self._xyzw_to_wxyz(target_tcp_quat_xyzw)
        q_slerp = quaternion_slerp(q_tcp, q_target, slerp_fraction)
        q_slerp_xyzw = self._wxyz_to_xyzw(q_slerp)

        return Pose(
            position=Point(
                x=float(blended_xyz[0]),
                y=float(blended_xyz[1]),
                z=float(blended_xyz[2]),
            ),
            orientation=Quaternion(
                x=float(q_slerp_xyzw[0]),
                y=float(q_slerp_xyzw[1]),
                z=float(q_slerp_xyzw[2]),
                w=float(q_slerp_xyzw[3]),
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
            self.get_logger().warn("No observation available for perception policy.")
            return False
        predicted_port_xyz = self._predict_port_xyz(obs, task)
        target_tcp_quat_xyzw = self._predict_target_quat_xyzw(obs, task)
        learned_plug_xyz = None
        if z_offset <= self._learned_plug_after_z:
            learned_plug_xyz = self._predict_plug_xyz(obs, task)
        pose = self._target_pose_from_prediction(
            obs,
            task,
            predicted_port_xyz,
            z_offset=z_offset,
            position_fraction=position_fraction,
            slerp_fraction=slerp_fraction,
            reset_xy_integrator=reset_xy_integrator,
            target_tcp_quat_xyzw=target_tcp_quat_xyzw,
            learned_plug_xyz=learned_plug_xyz,
        )
        self.set_pose_target(move_robot=move_robot, pose=pose)
        return True

    def _run_lateral_search(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        task: Task,
        z_offset: float,
        entry_z_offset: float | None = None,
    ) -> None:
        offsets = self._lateral_search_patterns.get(task.port_type, [])
        amplitude = self._lateral_search_amplitude_by_port_type.get(task.port_type, 0.0)
        if not offsets and amplitude <= 0.0:
            return

        if not offsets:
            offsets = [
                (0.0, 0.0, 0.0),
                (amplitude, 0.0, 0.0),
                (-amplitude, 0.0, 0.0),
                (0.0, amplitude, 0.0),
                (0.0, -amplitude, 0.0),
                (amplitude, amplitude, 0.0),
                (amplitude, -amplitude, 0.0),
                (-amplitude, amplitude, 0.0),
                (-amplitude, -amplitude, 0.0),
                (0.0, 0.0, 0.0),
            ]
        self.get_logger().info(
            "Running final lateral search: "
            f"points={len(offsets)}, hold_steps={self._lateral_search_hold_steps}, "
            f"entry_z={entry_z_offset}"
        )
        for dx, dy, dz in offsets:
            self._runtime_port_bias = np.array([dx, dy, dz], dtype=np.float32)
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
            if entry_z_offset is not None and np.isfinite(entry_z_offset):
                for _ in range(max(0, self._lateral_search_retract_steps)):
                    self._set_predicted_pose_target(
                        get_observation,
                        move_robot,
                        task,
                        z_offset=entry_z_offset,
                        reset_xy_integrator=True,
                    )
                    self.sleep_for(0.05)

                search_z = float(entry_z_offset)
                while search_z >= z_offset:
                    self._set_predicted_pose_target(
                        get_observation,
                        move_robot,
                        task,
                        z_offset=search_z,
                    )
                    search_z -= max(self._lateral_search_insert_step, 0.0005)
                    self.sleep_for(0.05)

            for _ in range(max(0, self._lateral_search_hold_steps)):
                self._set_predicted_pose_target(
                    get_observation,
                    move_robot,
                    task,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                )
                self.sleep_for(0.05)
        if not self._keep_lateral_search_bias:
            self._runtime_port_bias = np.zeros(3, dtype=np.float32)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"PerceptionGuidedPolicy.insert_cable() task: {task}")
        send_feedback("running perception-guided insertion")
        self._filtered_port_xyz = None
        self._frozen_port_xyz = None
        self._frozen_target_quat_xyzw = None
        self._filtered_plug_relative_xyz = None
        self._filtered_visual_correction = None
        self._runtime_port_bias = np.zeros(3, dtype=np.float32)
        self._prediction_count = 0
        self._debug_errors_mm = []
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

        final_z_offset = self._final_z_offset_by_port_type.get(task.port_type, -0.015)
        entry_search_z_offset = self._lateral_search_entry_z_by_port_type.get(
            task.port_type,
            float("nan"),
        )
        use_entry_search = (
            self._has_lateral_search(task.port_type)
            and np.isfinite(entry_search_z_offset)
            and entry_search_z_offset > final_z_offset
        )
        direct_z_offset = entry_search_z_offset if use_entry_search else final_z_offset
        self.get_logger().info(f"Using final z_offset: {final_z_offset:0.5}")
        while True:
            if z_offset < direct_z_offset:
                break
            z_offset -= self._insert_step
            if int((0.2 - z_offset) / self._insert_step) % 25 == 0:
                self.get_logger().info(f"z_offset: {z_offset:0.5}")
            self._set_predicted_pose_target(
                get_observation,
                move_robot,
                task,
                z_offset=z_offset,
            )
            self.sleep_for(0.05)

        if use_entry_search:
            z_offset = direct_z_offset

        self._run_lateral_search(
            get_observation,
            move_robot,
            task,
            final_z_offset,
            entry_z_offset=entry_search_z_offset if use_entry_search else None,
        )

        self.get_logger().info("Waiting for connector to stabilize...")
        for _ in range(0, 20):
            self._set_predicted_pose_target(
                get_observation,
                move_robot,
                task,
                z_offset=z_offset,
            )
            self.sleep_for(0.25)

        self.get_logger().info("PerceptionGuidedPolicy.insert_cable() exiting...")
        self._log_debug_summary()
        return True
