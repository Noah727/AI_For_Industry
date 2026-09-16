from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aic_task_interfaces.msg import Task

from aic_submission.PerceptionGuidedPolicy import PerceptionGuidedPolicy
from aic_submission.PlugAwarePolicy import PlugAwarePolicy
from aic_submission.perception.inference import RelativePortPoseInference


class HybridPerceptionPolicy(PlugAwarePolicy):
    """Use the best learned port XYZ model with learned plug tracking.

    The relative-port checkpoint has been much more stable for port position
    than the joint plug-aware checkpoint. The plug-aware checkpoint is still
    useful for estimating the current plug pose, especially once the cable
    flexes during descent. This policy fuses those two pieces and freezes the
    port estimate before the close-contact phase so late visual noise cannot
    chase the controller into the board.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        default_checkpoint = (
            Path(__file__).resolve().parents[1]
            / "runs"
            / "relative_port_pose_100ep"
            / "best_model.pt"
        )
        checkpoint_path = Path(
            os.environ.get("AIC_HYBRID_PORT_CHECKPOINT", str(default_checkpoint))
        )
        device = os.environ.get("AIC_HYBRID_PORT_DEVICE", "auto")
        image_stride = int(os.environ.get("AIC_HYBRID_PORT_IMAGE_STRIDE", "4"))
        self._port_predictor = RelativePortPoseInference(
            checkpoint_path,
            device=device,
            image_stride=image_stride,
        )
        self._hybrid_port_filter_alpha = float(
            os.environ.get("AIC_HYBRID_PORT_FILTER_ALPHA", "0.35")
        )
        self._hybrid_freeze_port_after = int(
            os.environ.get("AIC_HYBRID_FREEZE_PORT_AFTER", "100")
        )
        self._hybrid_port_quat_source = os.environ.get(
            "AIC_HYBRID_PORT_QUAT_SOURCE", "fixed"
        ).lower()

        self.get_logger().info(
            "Loaded hybrid relative-port checkpoint "
            f"{self._port_predictor.checkpoint_path} on {self._port_predictor.device}; "
            f"epoch={self._port_predictor.checkpoint_epoch}, "
            f"val_loss={self._port_predictor.checkpoint_val_loss}"
        )

    def _filter_relative_port_xyz(self, raw_port_xyz: np.ndarray, task: Task) -> np.ndarray:
        port_xyz = raw_port_xyz.astype(np.float32, copy=True)
        if self._clamp_port_xy and task.port_type in self.PORT_XY_LIMITS_BY_PORT_TYPE:
            x_limits, y_limits = self.PORT_XY_LIMITS_BY_PORT_TYPE[task.port_type]
            port_xyz[0] = np.clip(port_xyz[0], x_limits[0], x_limits[1])
            port_xyz[1] = np.clip(port_xyz[1], y_limits[0], y_limits[1])
        if self._clamp_port_z and task.port_type in self.PORT_Z_BY_PORT_TYPE:
            port_xyz[2] = self.PORT_Z_BY_PORT_TYPE[task.port_type]

        freeze_port = (
            self._hybrid_freeze_port_after > 0
            and self._prediction_count >= self._hybrid_freeze_port_after
            and self._filtered_port_xyz is not None
        )
        if freeze_port:
            return self._filtered_port_xyz.copy()

        if self._filtered_port_xyz is None:
            self._filtered_port_xyz = port_xyz
        else:
            alpha = float(np.clip(self._hybrid_port_filter_alpha, 0.0, 1.0))
            self._filtered_port_xyz = (
                alpha * port_xyz + (1.0 - alpha) * self._filtered_port_xyz
            ).astype(np.float32)
        return self._filtered_port_xyz.copy()

    def _filter_plug_pose(
        self,
        obs,
        plug_xyz: np.ndarray,
        plug_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tcp_xyz = self._tcp_xyz_from_observation(obs)
        plug_relative_xyz = plug_xyz.astype(np.float32) - tcp_xyz
        plug_alpha = float(np.clip(self._plug_filter_alpha, 0.0, 1.0))

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
        return (
            (tcp_xyz + self._filtered_plug_relative_xyz).astype(np.float32),
            self._filtered_plug_quat.copy(),
        )

    def _predict_pose(self, obs, task: Task):
        if self._predictor is None:
            raise RuntimeError("HybridPerceptionPolicy needs a plug-aware checkpoint.")

        learned_plug = self._predictor.predict(obs, task)
        raw_port_xyz = self._port_predictor.predict_port_xyz(obs, task)
        port_xyz = self._filter_relative_port_xyz(raw_port_xyz, task)

        if self._hybrid_port_quat_source == "plugaware":
            port_quat = learned_plug.port_quat_xyzw
        else:
            port_type = (
                task.port_type
                if task.port_type in PerceptionGuidedPolicy.PORT_QUAT_XYZW_BY_PORT_TYPE
                else task.plug_type
            )
            port_quat = PerceptionGuidedPolicy.PORT_QUAT_XYZW_BY_PORT_TYPE[port_type]
        port_quat = self._normalize_quat_xyzw(port_quat).astype(np.float32)

        plug_xyz, plug_quat = self._filter_plug_pose(
            obs,
            learned_plug.plug_xyz,
            learned_plug.plug_quat_xyzw,
        )

        self._prediction_count += 1
        if self._prediction_count % self._debug_log_every == 1:
            self.get_logger().info(
                "Hybrid pose: "
                f"port=({port_xyz[0]:.4f}, {port_xyz[1]:.4f}, {port_xyz[2]:.4f}), "
                f"plug=({plug_xyz[0]:.4f}, {plug_xyz[1]:.4f}, {plug_xyz[2]:.4f}), "
                f"quat_source={self._hybrid_port_quat_source}"
            )

        return port_xyz, port_quat, plug_xyz, plug_quat
