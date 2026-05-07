import argparse
import json
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.stability_metrics import (
    analyze_qpos_sequence,
    write_report,
)
from general_motion_retargeting.utils.xsens import load_xsens_file


LOWER_BODY_SMOOTH_JOINTS = (
    "waist_y_joint",
    "waist_x_joint",
    "waist_z_joint",
    "l_hip_y_joint",
    "l_hip_x_joint",
    "l_hip_z_joint",
    "l_knee_y_joint",
    "l_ankle_y_joint",
    "l_ankle_x_joint",
    "r_hip_y_joint",
    "r_hip_x_joint",
    "r_hip_z_joint",
    "r_knee_y_joint",
    "r_ankle_y_joint",
    "r_ankle_x_joint",
)


def estimate_ground_offset(retargeter: GMR, motion_frames):
    offset = np.inf
    for human_data in motion_frames:
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(
            human_data, retargeter.human_root_name, retargeter.human_scale_table
        )
        human_data = retargeter.offset_human_data(
            human_data, retargeter.pos_offsets1, retargeter.rot_offsets1
        )
        for pos, _quat in human_data.values():
            if pos[2] < offset:
                offset = pos[2]
    if not np.isfinite(offset):
        return 0.0
    return float(offset)


def get_smoothing_qpos_indices(model):
    indices = [0, 1, 2]
    for joint_name in LOWER_BODY_SMOOTH_JOINTS:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            indices.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(sorted(set(indices)), dtype=np.int32)


def smooth_qpos(qpos, prev_qpos, qpos_indices, alpha):
    if prev_qpos is None or alpha >= 1.0:
        return qpos.copy()
    smoothed = qpos.copy()
    smoothed[qpos_indices] = (
        alpha * qpos[qpos_indices] + (1.0 - alpha) * prev_qpos[qpos_indices]
    )
    return smoothed


def save_motion(path, qpos_list, fps):
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    root_pos = np.asarray([qpos[:3] for qpos in qpos_list])
    root_rot = np.asarray([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.asarray([qpos[7:] for qpos in qpos_list])
    motion_data = {
        "fps": int(fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with path.open("wb") as f:
        pickle.dump(motion_data, f)


def retarget_bvh(args):
    motion_frames, actual_human_height, frame_time = load_xsens_file(args)
    motion_fps = max(1, round(1 / frame_time))

    retargeter = GMR(
        src_human="bvh_xsens",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        verbose=False,
    )
    ground_offset = estimate_ground_offset(retargeter, motion_frames) - args.ground_clearance
    retargeter.set_ground_offset(ground_offset)
    print(f"Apply ground offset: {ground_offset:.4f} m")

    qpos_indices = get_smoothing_qpos_indices(retargeter.model)
    prev_qpos = None
    qpos_list = []

    for frame in tqdm(motion_frames, desc="Retargeting"):
        qpos = retargeter.retarget(frame)
        qpos = smooth_qpos(qpos, prev_qpos, qpos_indices, args.smoothing_alpha)
        prev_qpos = qpos.copy()
        qpos_list.append(qpos)

    return retargeter.model, qpos_list, motion_fps


def print_summary(summary):
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run plain BVH->ELF3 retargeting and output stability diagnostics."
    )
    parser.add_argument("--bvh_file", required=True)
    parser.add_argument("--robot", choices=["elf3"], default="elf3")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--json_path", default=None)
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--scale", default=0.01, type=float)
    parser.add_argument("--reset_to_zero", action="store_true", default=False)
    parser.add_argument("--start", default=None, type=int)
    parser.add_argument("--end", default=None, type=int)
    parser.add_argument("--bvh_format", choices=["3DSM"], default="3DSM")
    parser.add_argument("--ground_clearance", type=float, default=0.03)
    parser.add_argument("--smoothing_alpha", type=float, default=0.35)
    parser.add_argument("--support_height", type=float, default=0.08)
    args = parser.parse_args()

    model, qpos_list, fps = retarget_bvh(args)
    save_motion(args.save_path, qpos_list, fps)
    print(f"Saved motion: {args.save_path}")

    report = analyze_qpos_sequence(
        model,
        qpos_list,
        support_height=args.support_height,
    )
    report["summary"]["fps"] = fps
    report["summary"]["motion_path"] = str(args.save_path)

    json_path = args.json_path
    csv_path = args.csv_path
    if json_path is None and csv_path is None:
        motion_path = Path(args.save_path)
        json_path = motion_path.with_name(motion_path.stem + "_stability.json")
        csv_path = motion_path.with_name(motion_path.stem + "_stability.csv")
    write_report(report, json_path=json_path, csv_path=csv_path)
    print_summary(report["summary"])
    if json_path:
        print(f"Saved JSON report: {json_path}")
    if csv_path:
        print(f"Saved CSV report: {csv_path}")
