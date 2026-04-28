import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.xsens_vendor.BVHParser import BVHParser, Anim
import numpy as np
import json
from pathlib import Path


class OffsetManager:
    """Lightweight offset loader used at runtime without requiring PyQt."""

    channel_names = ["X", "Y", "Z"]

    def __init__(self, default_path="offsets.json"):
        self.default_path = Path(default_path)

    def load_offsets(self, path=None):
        path = Path(path) if path is not None else self.default_path
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def parse_to_window_format(self, joint_names, offsets_dict):
        offsets = {}
        for joint_idx, joint_name in enumerate(joint_names):
            joint_data = offsets_dict.get(
                joint_name, {"X": 0.0, "Y": 0.0, "Z": 0.0}
            )
            for channel_idx, channel_name in enumerate(self.channel_names):
                offsets[(joint_idx, channel_idx)] = joint_data.get(channel_name, 0.0)
        return offsets


def _get_frame_position(frame_data, body_names):
    for body_name in body_names:
        if body_name in frame_data:
            return np.asarray(frame_data[body_name][0], dtype=float)
    return None


def estimate_human_height(frames, default_height=1.75):
    """Estimate body height robustly across the motion sequence.

    Using only the last frame is brittle for dance / ballet sequences where the
    actor may end in a crouched or tip-toe pose. We instead aggregate per-frame
    head-to-foot heights and use a high percentile to approximate an upright
    frame while still ignoring outliers.
    """

    height_samples = []
    for frame_data in frames:
        head_pos = _get_frame_position(frame_data, ["Head_end_site", "Head"])
        foot_positions = [
            _get_frame_position(
                frame_data,
                [foot_name],
            )
            for foot_name in (
                "LeftToe_end_site",
                "RightToe_end_site",
                "LeftToe",
                "RightToe",
                "LeftFootMod",
                "RightFootMod",
                "LeftAnkle",
                "RightAnkle",
            )
        ]
        foot_z_values = [pos[2] for pos in foot_positions if pos is not None]
        if head_pos is None or not foot_z_values:
            continue

        sample = float(head_pos[2] - min(foot_z_values))
        if np.isfinite(sample) and 0.5 < sample < 3.0:
            height_samples.append(sample)

    if not height_samples:
        return float(default_height)

    return float(np.percentile(np.asarray(height_samples, dtype=float), 95))


def bvh_parse(args):
    parser = BVHParser(axis_order="zxy", scale=args.scale)
    with open(args.bvh_file, "r") as f:
        bvh_text = f.read()
    rotations, positions = parser.parse(
        bvh_text, start=args.start, end=args.end, reset_to_zero=args.reset_to_zero
    )
    offset_manager = OffsetManager(default_path="offsets.json")
    loaded_offsets = offset_manager.load_offsets()
    offsets = offset_manager.parse_to_window_format(parser.names, loaded_offsets)
    new_rotations = np.zeros_like(rotations)
    joint_offset = np.zeros((new_rotations.shape[1], 3))
    for i in range(new_rotations.shape[1]):
        for j in range(3):
            joint_offset[i, j] = offsets[(i, j)]
    new_rotations = rotations + joint_offset
    positions = np.copy(parser.positions)
    _quats, _positions, _offsets, _parents = parser._MOTION_data_post_processing(
        new_rotations, positions, reset_to_zero=True
    )
    print("MOTION_data_post_processing")
    anim = Anim(_quats, _positions, _offsets, _parents, parser.names)
    global_data = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    return anim, global_data, parser.frame_time


def load_xsens_file(args):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    anim, global_data, frame_time = bvh_parse(args)
    frames = []
    for frame in range(anim.pos.shape[0]):
        result = {}
        for i, bone in enumerate(anim.bones):
            orientation = global_data[0][frame, i]
            position = global_data[1][frame, i]
            result[bone] = (position, orientation)

        # Add modified foot pose
        # To make the config file more universal,
        # here the descriptions of the key points of the bvh file
        # that xsens may obtain are aligned with Lafan1
        if args.bvh_format == "3DSM":
            result["LeftFootMod"] = (
                np.array(
                    [
                        result["LeftAnkle"][0][0],
                        result["LeftAnkle"][0][1],
                        result["LeftAnkle"][0][2],
                        # result["LeftToe"][0][2],
                    ]
                ),
                result["LeftAnkle"][1],
                # result["LeftToe_end_site"][1],
            )
            result["RightFootMod"] = (
                np.array(
                    [
                        result["RightAnkle"][0][0],
                        result["RightAnkle"][0][1],
                        result["RightAnkle"][0][2],
                        # result["RightToe"][0][2],
                    ]
                ),
                result["RightAnkle"][1],
                # result["RightToe_end_site"][1],
            )

            # result["Spine2"] = result.pop("Chest4")

        frames.append(result)

    human_height = estimate_human_height(frames)
    return frames, human_height, frame_time
