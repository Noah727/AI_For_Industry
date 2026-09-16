from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def pose_xyz_from_transform(transform):
    return np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        dtype=np.float64,
    )


def pose_xyz_from_pose(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64, copy=True)
    norm = np.linalg.norm(quat)
    if norm > 1.0e-12:
        quat /= norm
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(transform) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    translation = transform.translation
    rotation = transform.rotation
    matrix[:3, :3] = quat_xyzw_to_matrix(
        np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)
    )
    matrix[:3, 3] = np.array(
        [translation.x, translation.y, translation.z], dtype=np.float64
    )
    return matrix


def build_global_transforms(last_tf: dict) -> dict[str, np.ndarray]:
    children = {child for _, child in last_tf.keys()}
    parents = {parent for parent, _ in last_tf.keys()}
    roots = sorted(parents - children)
    graph = defaultdict(list)
    for (parent, child), (_, transform) in last_tf.items():
        graph[parent].append((child, transform_to_matrix(transform.transform)))

    global_tf: dict[str, np.ndarray] = {}
    stack = [(root, np.eye(4, dtype=np.float64)) for root in roots]
    while stack:
        frame, matrix = stack.pop()
        if frame in global_tf:
            continue
        global_tf[frame] = matrix
        for child, child_matrix in graph.get(frame, []):
            stack.append((child, matrix @ child_matrix))
    return global_tf


def read_bag(uri: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_map = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    msg_types = {topic: get_message(type_name) for topic, type_name in type_map.items()}
    relevant_topics = {
        "/scoring/tf",
        "/tf",
        "/tf_static",
        "/aic_controller/pose_commands",
        "/aic_controller/controller_state",
        "/scoring/insertion_event",
    }

    last_tf = {}
    frame_counts = defaultdict(int)
    last_pose_command = None
    first_pose_command = None
    last_controller_pose = None
    first_controller_pose = None
    insertion_events = []

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic not in relevant_topics:
            continue
        if topic not in msg_types:
            continue
        msg = deserialize_message(data, msg_types[topic])
        if topic in ("/scoring/tf", "/tf", "/tf_static"):
            for transform in msg.transforms:
                key = (transform.header.frame_id, transform.child_frame_id)
                last_tf[key] = (timestamp, transform)
                frame_counts[key] += 1
        elif topic == "/aic_controller/pose_commands":
            if first_pose_command is None:
                first_pose_command = msg
            last_pose_command = msg
        elif topic == "/aic_controller/controller_state":
            if first_controller_pose is None:
                first_controller_pose = msg.tcp_pose
            last_controller_pose = msg.tcp_pose
        elif topic == "/scoring/insertion_event":
            insertion_events.append(str(msg.data))

    return {
        "last_tf": last_tf,
        "frame_counts": frame_counts,
        "first_pose_command": first_pose_command,
        "last_pose_command": last_pose_command,
        "first_controller_pose": first_controller_pose,
        "last_controller_pose": last_controller_pose,
        "insertion_events": insertion_events,
    }


def print_vector(label: str, values: np.ndarray) -> None:
    print(f"{label}: ({values[0]: .6f}, {values[1]: .6f}, {values[2]: .6f})")


def find_child(last_tf, fragments: list[str]):
    matches = []
    for (parent, child), (timestamp, transform) in last_tf.items():
        if all(fragment in child for fragment in fragments):
            matches.append((parent, child, timestamp, transform))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_dir", type=Path)
    args = parser.parse_args()

    data = read_bag(args.bag_dir)
    last_tf = data["last_tf"]
    global_tf = build_global_transforms(last_tf)

    print(f"bag: {args.bag_dir}")
    print(f"insertion_events: {data['insertion_events']}")
    if data["first_controller_pose"] is not None and data["last_controller_pose"] is not None:
        print_vector("first tcp", pose_xyz_from_pose(data["first_controller_pose"]))
        print_vector("last tcp ", pose_xyz_from_pose(data["last_controller_pose"]))

    if data["first_pose_command"] is not None:
        print("pose_command type:", type(data["first_pose_command"]).__name__)
        print("pose_command fields:", data["first_pose_command"].get_fields_and_field_types())

    interesting = [
        ("sfp_port_0", ["sfp_port_0"]),
        ("sfp_port_1", ["sfp_port_1"]),
        ("sc_port_base", ["sc_port_base"]),
        ("sfp_tip", ["sfp_tip"]),
        ("sc_tip", ["sc_tip"]),
    ]
    for label, fragments in interesting:
        matches = find_child(last_tf, fragments)
        if not matches:
            continue
        print(f"\n{label} matches:")
        for parent, child, _, transform in matches[:10]:
            print_vector(f"  {parent}->{child}", pose_xyz_from_transform(transform.transform))

    global_frames = [
        "task_board/nic_card_mount_0/sfp_port_0_link",
        "task_board/nic_card_mount_1/sfp_port_0_link",
        "task_board/sc_port_1/sc_port_base_link",
        "task_board/nic_card_mount_0/sfp_port_0_link_entrance",
        "task_board/nic_card_mount_1/sfp_port_0_link_entrance",
        "task_board/sc_port_1/sc_port_base_link_entrance",
        "cable_0/sfp_tip_link",
        "cable_0/sc_tip_link",
        "cable_1/sfp_tip_link",
        "cable_1/sc_tip_link",
    ]
    print("\nGlobal frame positions:")
    for frame in global_frames:
        if frame in global_tf:
            print_vector(f"  {frame}", global_tf[frame][:3, 3])

    if "base_link" in global_tf:
        base_inv = np.linalg.inv(global_tf["base_link"])
        print("\nBase-link frame positions:")
        for frame in global_frames:
            if frame in global_tf:
                local = base_inv @ global_tf[frame]
                print_vector(f"  {frame}", local[:3, 3])

    port_candidates = find_child(last_tf, ["port"])
    plug_candidates = find_child(last_tf, ["tip"])
    print("\nGlobal distances from matching plug tips to likely ports:")
    for _, plug_child, _, _ in plug_candidates:
        if plug_child not in global_tf:
            continue
        plug_xyz = global_tf[plug_child][:3, 3]
        for _, port_child, _, _ in port_candidates:
            if port_child not in global_tf:
                continue
            port_xyz = global_tf[port_child][:3, 3]
            if ("sfp" in plug_child and "sfp" in port_child) or (
                "sc" in plug_child and "sc" in port_child
            ):
                delta = plug_xyz - port_xyz
                print(
                    f"  {plug_child} -> {port_child}: "
                    f"dist={np.linalg.norm(delta):.5f} "
                    f"delta=({delta[0]:.5f}, {delta[1]:.5f}, {delta[2]:.5f})"
                )


if __name__ == "__main__":
    main()
