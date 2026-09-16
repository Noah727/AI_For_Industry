import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, Transform
from rclpy.time import Time
from tf2_ros import TransformException


class TeacherRecorderPolicy(CheatCode):
    """Run the CheatCode teacher while saving auto-labeled perception data.

    This policy is for training only. It requires the eval environment to run
    with `ground_truth:=true`, because labels are generated from TF.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._dataset_root = Path(
            os.environ.get("AIC_TEACHER_DATASET_DIR", "/tmp/aic_teacher_dataset")
        )
        self._sample_every = max(
            1, int(os.environ.get("AIC_RECORDER_SAMPLE_EVERY", "5"))
        )
        self._image_stride = max(
            1, int(os.environ.get("AIC_RECORDER_IMAGE_STRIDE", "4"))
        )
        self._episode_dir: Path | None = None
        self._sample_index = 0
        self._tick_index = 0

    @staticmethod
    def _pose_to_array(pose: Pose) -> np.ndarray:
        return np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _transform_to_array(transform: Transform) -> np.ndarray:
        return np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _safe_float_array(values, size: int | None = None) -> np.ndarray:
        array = np.asarray(list(values), dtype=np.float32)
        if size is not None:
            return array[:size]
        return array

    @staticmethod
    def _camera_info_payload(prefix: str, camera_info) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_camera_k": np.asarray(camera_info.k, dtype=np.float32).reshape(
                3, 3
            ),
            f"{prefix}_camera_p": np.asarray(camera_info.p, dtype=np.float32).reshape(
                3, 4
            ),
            f"{prefix}_camera_d": np.asarray(camera_info.d, dtype=np.float32),
            f"{prefix}_camera_width": np.asarray(camera_info.width, dtype=np.int32),
            f"{prefix}_camera_height": np.asarray(camera_info.height, dtype=np.int32),
            f"{prefix}_camera_frame_id": np.asarray(camera_info.header.frame_id),
            f"{prefix}_camera_distortion_model": np.asarray(
                camera_info.distortion_model
            ),
        }

    def _camera_pose_base(self, frame_id: str) -> np.ndarray:
        if not frame_id:
            return np.full(7, np.nan, dtype=np.float32)
        try:
            camera_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                frame_id,
                Time(),
            )
            return self._transform_to_array(camera_tf.transform)
        except TransformException as ex:
            self.get_logger().warn(
                f"Camera TF lookup failed for frame '{frame_id}': {ex}"
            )
            return np.full(7, np.nan, dtype=np.float32)

    def _image_to_array(self, image_msg) -> np.ndarray:
        data = np.frombuffer(image_msg.data, dtype=np.uint8)
        pixels = image_msg.height * image_msg.width
        if pixels <= 0:
            return np.zeros((0, 0, 0), dtype=np.uint8)

        channels = max(1, len(data) // pixels)
        expected = image_msg.height * image_msg.width * channels
        data = data[:expected]
        image = data.reshape(image_msg.height, image_msg.width, channels)
        return image[:: self._image_stride, :: self._image_stride].copy()

    def _state_from_observation(self, obs: Observation) -> np.ndarray:
        tcp_pose = obs.controller_state.tcp_pose
        tcp_vel = obs.controller_state.tcp_velocity
        return np.array(
            [
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                *obs.controller_state.tcp_error,
                *obs.joint_states.position[:7],
            ],
            dtype=np.float32,
        )

    def _wrench_from_observation(self, obs: Observation) -> np.ndarray:
        wrench = obs.wrist_wrench.wrench
        return np.array(
            [
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                wrench.torque.x,
                wrench.torque.y,
                wrench.torque.z,
            ],
            dtype=np.float32,
        )

    def _start_episode(self, task: Task) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_task_id = task.id if task.id else "task"
        safe_task_id = safe_task_id.replace("/", "_").replace(" ", "_")
        self._episode_dir = self._dataset_root / f"{timestamp}_{safe_task_id}"
        self._episode_dir.mkdir(parents=True, exist_ok=False)
        self._sample_index = 0
        self._tick_index = 0

        metadata = {
            "task": {
                "id": task.id,
                "cable_type": task.cable_type,
                "cable_name": task.cable_name,
                "plug_type": task.plug_type,
                "plug_name": task.plug_name,
                "port_type": task.port_type,
                "port_name": task.port_name,
                "target_module_name": task.target_module_name,
                "time_limit": int(task.time_limit),
            },
            "frames": {
                "port": f"task_board/{task.target_module_name}/{task.port_name}_link",
                "plug": f"{task.cable_name}/{task.plug_name}_link",
                "tcp": "gripper/tcp",
                "base": "base_link",
            },
            "recorder": {
                "sample_every": self._sample_every,
                "image_stride": self._image_stride,
                "image_layout": "height_width_channels_uint8",
                "pose_layout": "x_y_z_qx_qy_qz_qw",
                "state_layout": (
                    "tcp_xyz tcp_qxyzw tcp_linear_xyz tcp_angular_xyz "
                    "tcp_error_6 joint_positions_7"
                ),
            },
        }
        with open(self._episode_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.get_logger().info(f"Recording teacher dataset to {self._episode_dir}")

    def _record_sample(
        self,
        get_observation: GetObservationCallback,
        task: Task,
        port_transform: Transform,
        teacher_pose: Pose,
        stage: str,
        z_offset: float,
    ) -> None:
        self._tick_index += 1
        if self._tick_index % self._sample_every != 0:
            return
        if self._episode_dir is None:
            return

        obs = get_observation()
        if obs is None:
            self.get_logger().warn("Skipping recorder sample: no observation yet.")
            return

        try:
            plug_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                f"{task.cable_name}/{task.plug_name}_link",
                Time(),
            )
            tcp_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                "gripper/tcp",
                Time(),
            )
        except TransformException as ex:
            self.get_logger().warn(f"Skipping recorder sample: TF failed: {ex}")
            return

        sample_path = self._episode_dir / f"sample_{self._sample_index:06d}.npz"
        camera_payload = {}
        for prefix, camera_info in (
            ("left", obs.left_camera_info),
            ("center", obs.center_camera_info),
            ("right", obs.right_camera_info),
        ):
            camera_payload.update(self._camera_info_payload(prefix, camera_info))
            camera_payload[f"{prefix}_camera_pose_base"] = self._camera_pose_base(
                camera_info.header.frame_id
            )

        np.savez_compressed(
            sample_path,
            left_image=self._image_to_array(obs.left_image),
            center_image=self._image_to_array(obs.center_image),
            right_image=self._image_to_array(obs.right_image),
            state=self._state_from_observation(obs),
            wrench=self._wrench_from_observation(obs),
            joint_names=np.asarray(list(obs.joint_states.name), dtype=str),
            port_pose_base=self._transform_to_array(port_transform),
            plug_pose_base=self._transform_to_array(plug_tf.transform),
            tcp_pose_base=self._transform_to_array(tcp_tf.transform),
            teacher_tcp_target_base=self._pose_to_array(teacher_pose),
            stage=np.asarray(stage),
            z_offset=np.asarray(z_offset, dtype=np.float32),
            tick_index=np.asarray(self._tick_index, dtype=np.int64),
            image_stride=np.asarray(self._image_stride, dtype=np.int32),
            **camera_payload,
        )
        self._sample_index += 1

        if self._sample_index % 25 == 0:
            self.get_logger().info(
                f"Recorded {self._sample_index} samples in {self._episode_dir}"
            )

    def _finish_episode(self) -> None:
        if self._episode_dir is None:
            return
        summary_path = self._episode_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"num_samples": self._sample_index}, f, indent=2)
        self.get_logger().info(
            f"Teacher recording finished with {self._sample_index} samples: "
            f"{self._episode_dir}"
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"TeacherRecorderPolicy.insert_cable() task: {task}")
        self._task = task
        self._start_episode(task)

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False
        port_transform = port_tf_stamped.transform

        z_offset = 0.2

        for t in range(0, 100):
            interp_fraction = t / 100.0
            try:
                teacher_pose = self.calc_gripper_pose(
                    port_transform,
                    slerp_fraction=interp_fraction,
                    position_fraction=interp_fraction,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                )
                self._record_sample(
                    get_observation,
                    task,
                    port_transform,
                    teacher_pose,
                    stage="approach",
                    z_offset=z_offset,
                )
                self.set_pose_target(move_robot=move_robot, pose=teacher_pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(0.05)

        while True:
            if z_offset < -0.015:
                break

            z_offset -= 0.0005
            try:
                teacher_pose = self.calc_gripper_pose(port_transform, z_offset=z_offset)
                self._record_sample(
                    get_observation,
                    task,
                    port_transform,
                    teacher_pose,
                    stage="insert",
                    z_offset=z_offset,
                )
                self.set_pose_target(move_robot=move_robot, pose=teacher_pose)
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
            self.sleep_for(0.05)

        self.get_logger().info("Waiting for connector to stabilize...")
        for _ in range(0, 20):
            try:
                teacher_pose = self.calc_gripper_pose(port_transform, z_offset=z_offset)
                self._record_sample(
                    get_observation,
                    task,
                    port_transform,
                    teacher_pose,
                    stage="stabilize",
                    z_offset=z_offset,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during stabilization: {ex}")
            self.sleep_for(0.25)

        self._finish_episode()
        self.get_logger().info("TeacherRecorderPolicy.insert_cable() exiting...")
        return True
