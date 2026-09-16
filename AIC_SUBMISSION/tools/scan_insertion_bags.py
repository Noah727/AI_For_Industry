from __future__ import annotations

import argparse
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_events(bag_dir: Path) -> list[str]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if "/scoring/insertion_event" not in types:
        return []

    msg_type = get_message(types["/scoring/insertion_event"])
    events: list[str] = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/scoring/insertion_event":
            events.append(str(deserialize_message(data, msg_type).data))
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--glob", default="bag_trial_*")
    args = parser.parse_args()

    for bag_dir in sorted(args.root.glob(args.glob)):
        if not bag_dir.is_dir():
            continue
        metadata = bag_dir / "metadata.yaml"
        if not metadata.exists():
            continue
        events = read_events(bag_dir)
        if events:
            print(f"{bag_dir}: {events}")


if __name__ == "__main__":
    main()
