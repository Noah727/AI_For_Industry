from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aic_task_interfaces.msg import Task

from aic_submission.PlugAwarePolicy import PlugAwarePolicy
from aic_submission.perception.keypoint_inference import KeypointHeatmapInference


class KeypointServoPolicy(PlugAwarePolicy):
    """Plug-aware insertion with image-space keypoint alignment correction."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        if (
            "AIC_PLUG_AWARE_FREEZE_PORT_AFTER" not in os.environ
            and self._freeze_port_after == 0
        ):
            self._freeze_port_after = 1

        default_checkpoint = (
            Path(__file__).resolve().parents[1]
            / "runs"
            / "keypoint_center_100ep"
            / "best_model.pt"
        )
        checkpoint_path = Path(
            os.environ.get("AIC_KEYPOINT_CHECKPOINT", str(default_checkpoint))
        )
        device = os.environ.get("AIC_KEYPOINT_DEVICE", "auto")
        image_stride = int(os.environ.get("AIC_KEYPOINT_IMAGE_STRIDE", "4"))
        self._keypoint_predictor = KeypointHeatmapInference(
            checkpoint_path=checkpoint_path,
            device=device,
            image_key=os.environ.get("AIC_KEYPOINT_IMAGE_KEY", "center_image"),
            image_stride=image_stride,
        )
        self._keypoint_min_confidence = float(
            os.environ.get("AIC_KEYPOINT_MIN_CONFIDENCE", "0.45")
        )
        self._keypoint_gain = float(os.environ.get("AIC_KEYPOINT_CORRECTION_GAIN", "0.30"))
        self._keypoint_max_correction_m = float(
            os.environ.get("AIC_KEYPOINT_MAX_CORRECTION_M", "0.012")
        )
        self._keypoint_filter_alpha = float(
            os.environ.get("AIC_KEYPOINT_FILTER_ALPHA", "0.30")
        )
        self._keypoint_enable_after_count = int(
            os.environ.get("AIC_KEYPOINT_ENABLE_AFTER_COUNT", "0")
        )
        self._keypoint_update_min_z_offset = float(
            os.environ.get("AIC_KEYPOINT_UPDATE_MIN_Z_OFFSET", "0.050")
        )
        self._keypoint_max_delta_px = float(
            os.environ.get("AIC_KEYPOINT_MAX_DELTA_PX", "300")
        )
        self._keypoint_max_raw_correction_m = float(
            os.environ.get("AIC_KEYPOINT_MAX_RAW_CORRECTION_M", "0.025")
        )
        self._filtered_visual_correction: np.ndarray | None = None
        self._current_z_offset: float | None = None

        self.get_logger().info(
            "Loaded keypoint checkpoint "
            f"{self._keypoint_predictor.checkpoint_path} on "
            f"{self._keypoint_predictor.device}; "
            f"epoch={self._keypoint_predictor.checkpoint_epoch}, "
            f"val_px={self._keypoint_predictor.checkpoint_val_pixel_error}; "
            f"gain={self._keypoint_gain}, "
            f"max_correction={self._keypoint_max_correction_m:.3f}m, "
            f"min_conf={self._keypoint_min_confidence:.2f}, "
            f"max_delta={self._keypoint_max_delta_px:.0f}px"
        )

    @staticmethod
    def _clamp_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm > max_norm > 0.0:
            return vector * (max_norm / norm)
        return vector

    def _apply_visual_correction(self, port_xyz: np.ndarray) -> np.ndarray:
        if self._filtered_visual_correction is None:
            return port_xyz

        correction = self._keypoint_gain * self._filtered_visual_correction
        corrected = port_xyz.copy()
        corrected[:2] = corrected[:2] + correction[:2]
        return corrected.astype(np.float32)

    def _visual_corrected_port_xyz(
        self,
        obs,
        task: Task,
        port_xyz: np.ndarray,
        plug_xyz: np.ndarray,
    ) -> np.ndarray:
        if self._prediction_count < self._keypoint_enable_after_count:
            return self._apply_visual_correction(port_xyz)

        if (
            self._current_z_offset is not None
            and self._current_z_offset < self._keypoint_update_min_z_offset
        ):
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
        )

        alpha = float(np.clip(self._keypoint_filter_alpha, 0.0, 1.0))
        if self._filtered_visual_correction is None:
            self._filtered_visual_correction = desired_correction
        else:
            self._filtered_visual_correction = (
                alpha * desired_correction
                + (1.0 - alpha) * self._filtered_visual_correction
            ).astype(np.float32)

        corrected = self._apply_visual_correction(port_xyz)

        if self._clamp_port_xy and task.port_type in self.PORT_XY_LIMITS_BY_PORT_TYPE:
            x_limits, y_limits = self.PORT_XY_LIMITS_BY_PORT_TYPE[task.port_type]
            corrected[0] = np.clip(corrected[0], x_limits[0], x_limits[1])
            corrected[1] = np.clip(corrected[1], y_limits[0], y_limits[1])

        if self._prediction_count % self._debug_log_every == 0:
            delta_uv = prediction.delta_uv_full
            self.get_logger().info(
                "Keypoint servo: "
                f"duv=({delta_uv[0]:.1f}, {delta_uv[1]:.1f})px, "
                f"conf=({prediction.port_confidence:.2f}, "
                f"{prediction.plug_confidence:.2f}), "
                f"corr=({(corrected[0] - port_xyz[0])*1000.0:.1f}, "
                f"{(corrected[1] - port_xyz[1])*1000.0:.1f})mm"
            )
        return corrected.astype(np.float32)

    def _set_predicted_pose_target(
        self,
        get_observation,
        move_robot,
        task: Task,
        z_offset: float,
        position_fraction: float = 1.0,
        slerp_fraction: float = 1.0,
        reset_xy_integrator: bool = False,
    ) -> bool:
        self._current_z_offset = z_offset
        return super()._set_predicted_pose_target(
            get_observation=get_observation,
            move_robot=move_robot,
            task=task,
            z_offset=z_offset,
            position_fraction=position_fraction,
            slerp_fraction=slerp_fraction,
            reset_xy_integrator=reset_xy_integrator,
        )

    def _predict_pose(self, obs, task: Task):
        port_xyz, port_quat, plug_xyz, plug_quat = super()._predict_pose(obs, task)
        port_xyz = self._visual_corrected_port_xyz(obs, task, port_xyz, plug_xyz)
        return port_xyz, port_quat, plug_xyz, plug_quat

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self._filtered_visual_correction = None
        self._current_z_offset = None
        send_feedback("running keypoint-servo perception insertion")
        return super().insert_cable(task, get_observation, move_robot, send_feedback)
