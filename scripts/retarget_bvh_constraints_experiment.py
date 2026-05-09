import argparse
import csv
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import mujoco as mj
import numpy as np
from rich import print
from tqdm import tqdm

GMR_ROOT = Path(__file__).resolve().parents[1]
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.stability_metrics import (
    analyze_motion_file,
    analyze_qpos_sequence,
    write_report as write_stability_report,
)
from general_motion_retargeting.utils.xsens import load_xsens_file
from tools.motion_postprocess import (
    optimize_motion,
    quality_only,
    write_report as write_quality_report,
)


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


def retarget_options_for_mode(mode, motion_fps):
    enabled = {
        "baseline": set(),
        "velocity": {"velocity_limits"},
        "collision": {"collision_avoidance"},
        "support": {"support_foot"},
        "stability": {"stability"},
        "constraints": {
            "velocity_limits",
            "collision_avoidance",
            "support_foot",
            "stability",
        },
    }[mode]
    return {
        "velocity_limits": {"enabled": "velocity_limits" in enabled},
        "collision_avoidance": {"enabled": "collision_avoidance" in enabled},
        "support_foot": {
            "enabled": "support_foot" in enabled,
            "motion_fps": motion_fps,
        },
        "stability": {"enabled": "stability" in enabled},
    }


def save_motion(path, qpos_list, fps):
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
    return motion_data


def summarize_ik_velocity(history):
    if not history:
        return {}
    return {
        "ik_velocity_max": float(max(item.get("max", 0.0) for item in history)),
        "ik_velocity_p95_max": float(max(item.get("p95", 0.0) for item in history)),
        "ik_velocity_mean": float(np.mean([item.get("mean", 0.0) for item in history])),
    }


def run_mode(args, mode, frames, actual_human_height, motion_fps, layout):
    retargeter = GMR(
        src_human="bvh_xsens",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        verbose=False,
        retarget_options=retarget_options_for_mode(mode, motion_fps),
    )
    ground_offset = estimate_ground_offset(retargeter, frames) - args.ground_clearance
    retargeter.set_ground_offset(ground_offset)

    qpos_indices = get_smoothing_qpos_indices(retargeter.model)
    prev_qpos = None
    qpos_list = []
    for frame in tqdm(frames, desc=f"Retarget {mode}"):
        qpos = retargeter.retarget(frame)
        qpos = smooth_qpos(qpos, prev_qpos, qpos_indices, args.smoothing_alpha)
        prev_qpos = qpos.copy()
        qpos_list.append(qpos)

    motion_path = layout["motions"] / f"robot_motion_{mode}.pkl"
    motion = save_motion(motion_path, qpos_list, motion_fps)

    quality = quality_only(motion, robot=args.robot)
    quality_path = layout["quality"] / f"quality_{mode}.json"
    write_quality_report(quality_path, quality)

    stability = analyze_qpos_sequence(
        retargeter.model,
        qpos_list,
        support_height=args.support_height,
    )
    stability["summary"]["fps"] = motion_fps
    stability["summary"]["motion_path"] = str(motion_path)
    stability_json_path = layout["stability"] / f"stability_{mode}.json"
    stability_csv_path = layout["stability"] / f"stability_{mode}.csv"
    write_stability_report(
        stability,
        json_path=stability_json_path,
        csv_path=stability_csv_path,
    )

    return {
        "mode": mode,
        "motion_path": str(motion_path),
        "quality_path": str(quality_path),
        "stability_json_path": str(stability_json_path),
        "stability_csv_path": str(stability_csv_path),
        "collision_fallback_failures": retargeter.collision_solve_failures,
        "ik_velocity_summary": summarize_ik_velocity(
            retargeter.ik_velocity_stats_history
        ),
        "quality": quality,
        "stability": stability,
    }


def run_postprocess_stage(args, source_result, layout):
    source_mode = source_result["mode"]
    stage_name = (
        f"{source_mode}_post_{args.postprocess_profile}_"
        f"{args.postprocess_pipeline}"
    )
    motion_path = layout["motions"] / f"robot_motion_{stage_name}.pkl"
    quality_path = layout["quality"] / f"quality_{stage_name}.json"
    stability_json_path = layout["stability"] / f"stability_{stage_name}.json"
    stability_csv_path = layout["stability"] / f"stability_{stage_name}.csv"

    with open(source_result["motion_path"], "rb") as f:
        source_motion = pickle.load(f)
    optimized_motion, quality = optimize_motion(
        source_motion,
        robot=args.robot,
        profile_name=args.postprocess_profile,
        pipeline_name=args.postprocess_pipeline,
    )
    quality["input"] = source_result["motion_path"]
    quality["output"] = str(motion_path)

    with motion_path.open("wb") as f:
        pickle.dump(optimized_motion, f)
    write_quality_report(quality_path, quality)

    stability = analyze_motion_file(
        args.robot,
        motion_path,
        root_rot_format="xyzw",
        support_height=args.support_height,
    )
    write_stability_report(
        stability,
        json_path=stability_json_path,
        csv_path=stability_csv_path,
    )

    return {
        "mode": stage_name,
        "stage_label": f"{source_mode} + postprocess",
        "motion_path": str(motion_path),
        "quality_path": str(quality_path),
        "stability_json_path": str(stability_json_path),
        "stability_csv_path": str(stability_csv_path),
        "collision_fallback_failures": "",
        "ik_velocity_summary": {},
        "quality": {"metrics": quality.get("after", quality.get("metrics", {}))},
        "stability": stability,
    }


def nested_get(data, keys, default=None):
    cursor = data
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def summary_row(result):
    metrics = result["quality"].get("metrics", {})
    stability = result["stability"].get("summary", {})
    return {
        "mode": result["mode"],
        "stage_label": result.get("stage_label", result["mode"]),
        "motion_path": result["motion_path"],
        "self_collision_frame_ratio": nested_get(
            metrics, ["self_collision", "collision_frame_ratio"], ""
        ),
        "self_collision_max_penetration_m": nested_get(
            metrics, ["self_collision", "max_penetration_m"], ""
        ),
        "foot_sliding_speed_p95": nested_get(
            metrics, ["contact", "estimated_foot_sliding_speed", "p95"], ""
        ),
        "dof_velocity_max": nested_get(metrics, ["dof_velocity", "max"], ""),
        "dof_acceleration_max": nested_get(metrics, ["dof_acceleration", "max"], ""),
        "dof_jerk_max": nested_get(metrics, ["dof_jerk", "max"], ""),
        "outside_support_percent": stability.get("outside_support_percent", ""),
        "min_support_margin_m": stability.get("min_support_margin_m", ""),
        "ik_velocity_max": result["ik_velocity_summary"].get("ik_velocity_max", ""),
        "ik_velocity_p95_max": result["ik_velocity_summary"].get(
            "ik_velocity_p95_max", ""
        ),
        "collision_fallback_failures": result["collision_fallback_failures"],
    }


def write_summary_csv(path, results):
    rows = [summary_row(result) for result in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_layout(output_dir):
    layout = {
        "root": output_dir,
        "motions": output_dir / "motions",
        "quality": output_dir / "quality",
        "stability": output_dir / "stability",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run BVH->ELF3 source-level GMR constraint experiments without opening "
            "the MuJoCo viewer."
        )
    )
    parser.add_argument("--bvh_file", required=True)
    parser.add_argument("--robot", choices=["elf3"], default="elf3")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "constraints"],
        choices=["baseline", "velocity", "collision", "support", "stability", "constraints"],
    )
    parser.add_argument("--scale", default=0.01, type=float)
    parser.add_argument("--reset_to_zero", action="store_true", default=False)
    parser.add_argument("--start", default=None, type=int)
    parser.add_argument("--end", default=None, type=int)
    parser.add_argument("--bvh_format", choices=["3DSM"], default="3DSM")
    parser.add_argument("--ground_clearance", type=float, default=0.03)
    parser.add_argument("--smoothing_alpha", type=float, default=0.35)
    parser.add_argument("--support_height", type=float, default=0.08)
    parser.add_argument(
        "--postprocess_modes",
        nargs="*",
        default=[],
        help=(
            "Retarget modes that should also be passed through the original "
            "motion_postprocess optimizer and included as extra summary stages."
        ),
    )
    parser.add_argument(
        "--postprocess_profile",
        default="soft",
        choices=["preview", "soft", "strict"],
    )
    parser.add_argument(
        "--postprocess_pipeline",
        default="v2_foot",
        choices=["legacy", "v2", "v2_foot", "collision"],
    )
    args = parser.parse_args()

    frames, actual_human_height, frame_time = load_xsens_file(args)
    motion_fps = max(1, round(1 / frame_time))

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path("runtime/experiments")
            / "constraints"
            / f"gmr_constraints_{stamp}"
        )
        output_dir = output_dir.resolve()
    layout = prepare_output_layout(output_dir)

    meta = {
        "bvh_file": str(Path(args.bvh_file).expanduser()),
        "robot": args.robot,
        "modes": args.modes,
        "frames": len(frames),
        "fps": motion_fps,
        "ground_clearance": args.ground_clearance,
        "smoothing_alpha": args.smoothing_alpha,
    }
    (output_dir / "experiment_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    results = []
    results_by_mode = {}
    for mode in args.modes:
        result = run_mode(args, mode, frames, actual_human_height, motion_fps, layout)
        result["stage_label"] = mode
        results.append(result)
        results_by_mode[mode] = result

    for mode in args.postprocess_modes:
        if mode not in results_by_mode:
            raise ValueError(f"Cannot postprocess missing mode: {mode}")
        results.append(run_postprocess_stage(args, results_by_mode[mode], layout))

    summary_path = output_dir / "summary.csv"
    write_summary_csv(summary_path, results)
    print(f"[OK] Saved experiment outputs: {output_dir}")
    print(f"[OK] Saved summary: {summary_path}")
