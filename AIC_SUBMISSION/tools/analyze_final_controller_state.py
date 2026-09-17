from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


DEFAULT_PORTS = {
    "trial_1": np.array([-0.384359, 0.212866, 0.133476], dtype=np.float64),
    "trial_2": np.array([-0.384332, 0.252866, 0.133476], dtype=np.float64),
    "trial_3": np.array([-0.488578, 0.288429, 0.014500], dtype=np.float64),
}

TCP_TO_PLUG = {
    "trial_1": np.array([0.000030, -0.020685, 0.054132], dtype=np.float64),
    "trial_2": np.array([-0.000370, -0.027075, 0.043190], dtype=np.float64),
    "trial_3": np.array([-0.001465, -0.009721, 0.005261], dtype=np.float64),
}


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64)
    quat /= np.linalg.norm(quat)
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_xyz(pose) -> np.ndarray:
    return np.array(
        [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
    )


def pose_quat(pose) -> np.ndarray:
    return np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )


def read_final(bag_dir: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    controller_type = get_message(types["/aic_controller/controller_state"])
    command_type = get_message(types["/aic_controller/pose_commands"])
    event_type = (
        get_message(types["/scoring/insertion_event"])
        if "/scoring/insertion_event" in types
        else None
    )

    first_tcp = None
    last_tcp = None
    first_command = None
    last_command = None
    events = []
    relevant_topics = {
        "/aic_controller/controller_state",
        "/aic_controller/pose_commands",
        "/scoring/insertion_event",
    }
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in relevant_topics:
            continue
        if topic == "/aic_controller/controller_state":
            msg = deserialize_message(data, controller_type)
            if first_tcp is None:
                first_tcp = msg.tcp_pose
            last_tcp = msg.tcp_pose
        elif topic == "/aic_controller/pose_commands":
            msg = deserialize_message(data, command_type)
            if first_command is None:
                first_command = msg.pose
            last_command = msg.pose
        elif topic == "/scoring/insertion_event" and event_type is not None:
            msg = deserialize_message(data, event_type)
            events.append(str(msg.data))
    return first_tcp, last_tcp, first_command, last_command, events


def fmt(values: np.ndarray) -> str:
    return f"({values[0]: .6f}, {values[1]: .6f}, {values[2]: .6f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("--trial", required=True, choices=sorted(DEFAULT_PORTS))
    args = parser.parse_args()

    first_tcp, last_tcp, first_command, last_command, events = read_final(args.bag_dir)
    port = DEFAULT_PORTS[args.trial]
    rel = TCP_TO_PLUG[args.trial]

    tcp_xyz = pose_xyz(last_tcp)
    tcp_quat = pose_quat(last_tcp)
    plug_xyz = tcp_xyz + quat_xyzw_to_matrix(tcp_quat).dot(rel)
    delta = plug_xyz - port

    cmd_xyz = pose_xyz(last_command)
    cmd_quat = pose_quat(last_command)
    cmd_plug_xyz = cmd_xyz + quat_xyzw_to_matrix(cmd_quat).dot(rel)
    cmd_delta = cmd_plug_xyz - port

    print(f"bag: {args.bag_dir}")
    print(f"events: {events}")
    print(f"first_tcp_xyz: {fmt(pose_xyz(first_tcp))}")
    print(f"last_tcp_xyz:  {fmt(tcp_xyz)}")
    print(
        f"last_tcp_qxyzw: ({tcp_quat[0]: .6f}, {tcp_quat[1]: .6f}, {tcp_quat[2]: .6f}, {tcp_quat[3]: .6f})"
    )
    print(f"last_cmd_xyz:  {fmt(cmd_xyz)}")
    print(f"port_xyz:      {fmt(port)}")
    print(f"actual_plug:   {fmt(plug_xyz)}")
    print(f"actual_delta:  {fmt(delta)} dist={np.linalg.norm(delta):.6f}")
    print(f"cmd_plug:      {fmt(cmd_plug_xyz)}")
    print(f"cmd_delta:     {fmt(cmd_delta)} dist={np.linalg.norm(cmd_delta):.6f}")


if __name__ == "__main__":
    main()
