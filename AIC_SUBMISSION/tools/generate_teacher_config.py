#!/usr/bin/env python3
"""Generate randomized aic_engine configs for teacher-data collection."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import yaml


def uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    return rng.uniform(bounds[0], bounds[1])


def false_entity() -> dict[str, Any]:
    return {"entity_present": False}


def randomize_board_pose(
    rng: random.Random,
    task_board: dict[str, Any],
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    yaw_bounds: tuple[float, float],
) -> None:
    pose = task_board["pose"]
    pose["x"] = round(uniform(rng, x_bounds), 5)
    pose["y"] = round(uniform(rng, y_bounds), 5)
    pose["yaw"] = round(uniform(rng, yaw_bounds), 5)


def randomize_cable_grasp(
    rng: random.Random,
    cable_cfg: dict[str, Any],
    *,
    z_jitter: float = 0.002,
    rpy_jitter: float = 0.04,
) -> None:
    pose = cable_cfg["pose"]
    pose["gripper_offset"]["z"] = round(
        pose["gripper_offset"]["z"] + uniform(rng, (-z_jitter, z_jitter)),
        6,
    )
    pose["roll"] = round(pose["roll"] + uniform(rng, (-rpy_jitter, rpy_jitter)), 6)
    pose["pitch"] = round(pose["pitch"] + uniform(rng, (-rpy_jitter, rpy_jitter)), 6)
    pose["yaw"] = round(pose["yaw"] + uniform(rng, (-rpy_jitter, rpy_jitter)), 6)


def make_sfp_trial(
    rng: random.Random,
    template: dict[str, Any],
    limits: dict[str, Any],
    trial_index: int,
) -> dict[str, Any]:
    trial = copy.deepcopy(template)
    board = trial["scene"]["task_board"]

    randomize_board_pose(
        rng,
        board,
        x_bounds=(0.12, 0.20),
        y_bounds=(-0.24, -0.16),
        yaw_bounds=(2.98, 3.22),
    )

    rail = rng.randrange(5)
    nic_limits = limits["nic_rail"]
    for idx in range(5):
        key = f"nic_rail_{idx}"
        board[key] = false_entity()
    board[f"nic_rail_{rail}"] = {
        "entity_present": True,
        "entity_name": f"nic_card_{rail}",
        "entity_pose": {
            "translation": round(
                uniform(
                    rng,
                    (
                        float(nic_limits["min_translation"]),
                        float(nic_limits["max_translation"]),
                    ),
                ),
                6,
            ),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": round(uniform(rng, (-0.08, 0.08)), 6),
        },
    }

    task = trial["tasks"]["task_1"]
    task["id"] = f"sfp_{trial_index:04d}"
    task["cable_name"] = "cable_0"
    task["plug_type"] = "sfp"
    task["plug_name"] = "sfp_tip"
    task["port_type"] = "sfp"
    task["port_name"] = rng.choice(["sfp_port_0", "sfp_port_1"])
    task["target_module_name"] = f"nic_card_mount_{rail}"

    randomize_cable_grasp(rng, trial["scene"]["cables"]["cable_0"])
    return trial


def make_sc_trial(
    rng: random.Random,
    template: dict[str, Any],
    limits: dict[str, Any],
    trial_index: int,
) -> dict[str, Any]:
    trial = copy.deepcopy(template)
    board = trial["scene"]["task_board"]

    randomize_board_pose(
        rng,
        board,
        x_bounds=(0.14, 0.20),
        y_bounds=(-0.04, 0.04),
        yaw_bounds=(2.88, 3.18),
    )

    rail = rng.randrange(2)
    sc_limits = limits["sc_rail"]
    for idx in range(2):
        key = f"sc_rail_{idx}"
        board[key] = false_entity()
    board[f"sc_rail_{rail}"] = {
        "entity_present": True,
        "entity_name": f"sc_mount_{rail}",
        "entity_pose": {
            "translation": round(
                uniform(
                    rng,
                    (
                        float(sc_limits["min_translation"]),
                        float(sc_limits["max_translation"]),
                    ),
                ),
                6,
            ),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },
    }

    task = trial["tasks"]["task_1"]
    task["id"] = f"sc_{trial_index:04d}"
    task["cable_name"] = "cable_1"
    task["plug_type"] = "sc"
    task["plug_name"] = "sc_tip"
    task["port_type"] = "sc"
    task["port_name"] = "sc_port_base"
    task["target_module_name"] = f"sc_port_{rail}"

    randomize_cable_grasp(rng, trial["scene"]["cables"]["cable_1"])
    return trial


def generate_config(
    sample_config_path: Path,
    output_path: Path,
    *,
    num_sfp: int,
    num_sc: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    with open(sample_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_sfp = config["trials"]["trial_1"]
    base_sc = config["trials"]["trial_3"]
    limits = config["task_board_limits"]

    trials: dict[str, Any] = {}
    trial_number = 1
    for idx in range(num_sfp):
        trials[f"trial_{trial_number}"] = make_sfp_trial(
            rng, base_sfp, limits, trial_number
        )
        trial_number += 1
    for idx in range(num_sc):
        trials[f"trial_{trial_number}"] = make_sc_trial(
            rng, base_sc, limits, trial_number
        )
        trial_number += 1

    config["trials"] = trials

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(
        f"Wrote {len(trials)} trials "
        f"({num_sfp} sfp, {num_sc} sc) to {output_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-config",
        type=Path,
        default=Path("aic_engine/config/sample_config.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("AIC_SUBMISSION/generated_configs/randomized_teacher.yaml"),
    )
    parser.add_argument("--num-sfp", type=int, default=20)
    parser.add_argument("--num-sc", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    generate_config(
        args.sample_config,
        args.output,
        num_sfp=args.num_sfp,
        num_sc=args.num_sc,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
