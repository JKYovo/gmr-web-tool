import csv
import json
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from .params import ROBOT_XML_DICT


FOOT_SPECS = {
    "left": {"body": "l_ankle_x_link", "site": "lf_tc"},
    "right": {"body": "r_ankle_x_link", "site": "rf_tc"},
}


def convex_hull_xy(points):
    points = sorted(set((float(point[0]), float(point[1])) for point in points))
    if len(points) <= 1:
        return np.asarray(points, dtype=float)

    def cross(origin, a, b):
        return (
            (a[0] - origin[0]) * (b[1] - origin[1])
            - (a[1] - origin[1]) * (b[0] - origin[0])
        )

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def ground_yaw_axes(body_mat):
    x_axis = body_mat[:, 0].copy()
    x_axis[2] = 0.0
    norm = np.linalg.norm(x_axis)
    if norm < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis /= norm
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-6)
    return x_axis, y_axis


def support_foot_corners(center, yaw_x, yaw_y, half_length=0.11, half_width=0.04):
    return np.array(
        [
            center + half_length * yaw_x + half_width * yaw_y,
            center + half_length * yaw_x - half_width * yaw_y,
            center - half_length * yaw_x - half_width * yaw_y,
            center - half_length * yaw_x + half_width * yaw_y,
        ],
        dtype=float,
    )


def signed_margin_to_convex_polygon(point, polygon):
    polygon = np.asarray(polygon, dtype=float)
    point = np.asarray(point, dtype=float)
    if len(polygon) < 3:
        return np.nan

    margins = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        edge_norm = np.linalg.norm(edge)
        if edge_norm < 1e-9:
            continue
        cross = edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0])
        margins.append(cross / edge_norm)
    if not margins:
        return np.nan
    return float(np.min(margins))


def normalize_quat(quat):
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def root_quat_to_wxyz(root_quat, root_rot_format):
    quat = normalize_quat(root_quat)
    if root_rot_format == "xyzw":
        return quat[[3, 0, 1, 2]]
    if root_rot_format == "wxyz":
        return quat
    raise ValueError(f"Unsupported root_rot_format: {root_rot_format}")


def qpos_from_motion_frame(root_pos, root_rot, dof_pos, root_rot_format="xyzw"):
    qpos = np.zeros(7 + len(dof_pos), dtype=float)
    qpos[:3] = root_pos
    qpos[3:7] = root_quat_to_wxyz(root_rot, root_rot_format)
    qpos[7:] = dof_pos
    return qpos


def load_motion_arrays(motion_path):
    with Path(motion_path).open("rb") as f:
        motion = pickle.load(f)
    return (
        int(motion["fps"]),
        np.asarray(motion["root_pos"], dtype=float),
        np.asarray(motion["root_rot"], dtype=float),
        np.asarray(motion["dof_pos"], dtype=float),
    )


def joint_qpos_value(model, data, joint_name):
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return np.nan
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def body_rotation(model, data, body_name):
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    return data.xmat[body_id].reshape(3, 3)


def body_position(model, data, body_name):
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    return data.xpos[body_id].copy()


def lean_angles_deg(body_mat):
    yaw_x, yaw_y = ground_yaw_axes(body_mat)
    body_z = body_mat[:, 2]
    forward = np.degrees(np.arctan2(np.dot(body_z, yaw_x), body_z[2]))
    left = np.degrees(np.arctan2(np.dot(body_z, yaw_y), body_z[2]))
    return float(forward), float(left)


def euler_xyz_deg(body_mat):
    return R.from_matrix(body_mat).as_euler("xyz", degrees=True)


def support_state(model, data, support_height=0.08):
    feet = {}
    for side, spec in FOOT_SPECS.items():
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, spec["site"])
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, spec["body"])
        if site_id < 0 or body_id < 0:
            continue
        body_mat = data.xmat[body_id].reshape(3, 3)
        yaw_x, yaw_y = ground_yaw_axes(body_mat)
        center = data.xpos[body_id].copy()
        center[2] = 0.0
        feet[side] = {
            "site": data.site_xpos[site_id].copy(),
            "center": center,
            "yaw_x": yaw_x,
            "yaw_y": yaw_y,
        }

    if not feet:
        return {}, [], np.empty((0, 2)), np.full(2, np.nan)

    min_z = min(float(foot["site"][2]) for foot in feet.values())
    support_sides = [
        side for side, foot in feet.items() if float(foot["site"][2]) <= min_z + support_height
    ]
    if not support_sides:
        support_sides = [min(feet, key=lambda side: float(feet[side]["site"][2]))]

    corners = []
    centers = []
    for side in support_sides:
        foot = feet[side]
        centers.append(foot["site"][:2])
        foot_corners = support_foot_corners(
            foot["center"][:2],
            foot["yaw_x"][:2],
            foot["yaw_y"][:2],
        )
        corners.extend(foot_corners)
    hull = convex_hull_xy(corners)
    support_center = np.mean(np.asarray(centers, dtype=float), axis=0)
    return feet, support_sides, hull, support_center


def compute_frame_metrics(model, data, frame_idx, support_height=0.08):
    torso_mat = body_rotation(model, data, "torso_link")
    waist_mat = body_rotation(model, data, "waist_z_link")
    torso_pos = body_position(model, data, "torso_link")
    waist_pos = body_position(model, data, "waist_z_link")

    torso_forward_lean, torso_left_lean = (
        lean_angles_deg(torso_mat) if torso_mat is not None else (np.nan, np.nan)
    )
    waist_forward_lean, waist_left_lean = (
        lean_angles_deg(waist_mat) if waist_mat is not None else (np.nan, np.nan)
    )
    torso_euler = euler_xyz_deg(torso_mat) if torso_mat is not None else np.full(3, np.nan)
    waist_euler = euler_xyz_deg(waist_mat) if waist_mat is not None else np.full(3, np.nan)

    feet, support_sides, hull, support_center = support_state(
        model, data, support_height=support_height
    )
    com = data.subtree_com[1].copy()
    support_margin = signed_margin_to_convex_polygon(com[:2], hull)

    if torso_mat is not None and np.all(np.isfinite(support_center)):
        yaw_x, yaw_y = ground_yaw_axes(torso_mat)
        com_delta = com[:2] - support_center
        waist_delta = (
            waist_pos[:2] - support_center if waist_pos is not None else np.full(2, np.nan)
        )
        torso_delta = (
            torso_pos[:2] - support_center if torso_pos is not None else np.full(2, np.nan)
        )
        com_forward = float(np.dot(com_delta, yaw_x[:2]))
        com_left = float(np.dot(com_delta, yaw_y[:2]))
        waist_forward = float(np.dot(waist_delta, yaw_x[:2]))
        torso_forward = float(np.dot(torso_delta, yaw_x[:2]))
    else:
        com_forward = com_left = waist_forward = torso_forward = np.nan

    return {
        "frame": int(frame_idx),
        "support": "".join(side[0].upper() for side in support_sides),
        "support_margin_m": support_margin,
        "com_x_m": float(com[0]),
        "com_y_m": float(com[1]),
        "com_z_m": float(com[2]),
        "com_forward_from_support_m": com_forward,
        "com_left_from_support_m": com_left,
        "torso_forward_from_support_m": torso_forward,
        "waist_forward_from_support_m": waist_forward,
        "torso_forward_lean_deg": torso_forward_lean,
        "torso_left_lean_deg": torso_left_lean,
        "waist_forward_lean_deg": waist_forward_lean,
        "waist_left_lean_deg": waist_left_lean,
        "torso_euler_x_deg": float(torso_euler[0]),
        "torso_euler_y_deg": float(torso_euler[1]),
        "torso_euler_z_deg": float(torso_euler[2]),
        "waist_euler_x_deg": float(waist_euler[0]),
        "waist_euler_y_deg": float(waist_euler[1]),
        "waist_euler_z_deg": float(waist_euler[2]),
        "waist_y_joint_rad": joint_qpos_value(model, data, "waist_y_joint"),
        "waist_x_joint_rad": joint_qpos_value(model, data, "waist_x_joint"),
        "l_ankle_y_joint_rad": joint_qpos_value(model, data, "l_ankle_y_joint"),
        "r_ankle_y_joint_rad": joint_qpos_value(model, data, "r_ankle_y_joint"),
        "left_foot_z_m": float(feet["left"]["site"][2]) if "left" in feet else np.nan,
        "right_foot_z_m": float(feet["right"]["site"][2]) if "right" in feet else np.nan,
    }


def summarize_rows(rows):
    if not rows:
        return {}

    def values(key):
        return np.asarray(
            [row[key] for row in rows if np.isfinite(row.get(key, np.nan))],
            dtype=float,
        )

    def mean(key):
        vals = values(key)
        return float(np.mean(vals)) if len(vals) else np.nan

    def max_abs(key):
        vals = values(key)
        return float(np.max(np.abs(vals))) if len(vals) else np.nan

    def p95_abs(key):
        vals = values(key)
        return float(np.percentile(np.abs(vals), 95)) if len(vals) else np.nan

    margins = values("support_margin_m")
    supports = [row["support"] for row in rows]
    return {
        "frames": len(rows),
        "outside_support_percent": (
            float(np.mean(margins < -1e-6) * 100.0) if len(margins) else np.nan
        ),
        "min_support_margin_m": float(np.min(margins)) if len(margins) else np.nan,
        "mean_support_margin_m": float(np.mean(margins)) if len(margins) else np.nan,
        "mean_com_forward_from_support_m": mean("com_forward_from_support_m"),
        "p95_abs_com_forward_from_support_m": p95_abs("com_forward_from_support_m"),
        "max_abs_torso_forward_lean_deg": max_abs("torso_forward_lean_deg"),
        "p95_abs_torso_forward_lean_deg": p95_abs("torso_forward_lean_deg"),
        "max_abs_waist_forward_lean_deg": max_abs("waist_forward_lean_deg"),
        "p95_abs_waist_forward_lean_deg": p95_abs("waist_forward_lean_deg"),
        "mean_waist_forward_from_support_m": mean("waist_forward_from_support_m"),
        "single_support_percent": float(
            np.mean([support in ("L", "R") for support in supports]) * 100.0
        ),
    }


def analyze_qpos_sequence(model, qpos_sequence, support_height=0.08):
    data = mj.MjData(model)
    rows = []
    for frame_idx, qpos in enumerate(qpos_sequence):
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        rows.append(
            compute_frame_metrics(
                model,
                data,
                frame_idx=frame_idx,
                support_height=support_height,
            )
        )
    return {"summary": summarize_rows(rows), "frames": rows}


def analyze_motion_file(robot, motion_path, root_rot_format="xyzw", support_height=0.08):
    fps, root_pos, root_rot, dof_pos = load_motion_arrays(motion_path)
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    qpos_sequence = [
        qpos_from_motion_frame(root_pos[idx], root_rot[idx], dof_pos[idx], root_rot_format)
        for idx in range(len(root_pos))
    ]
    report = analyze_qpos_sequence(model, qpos_sequence, support_height=support_height)
    report["summary"]["fps"] = fps
    report["summary"]["motion_path"] = str(motion_path)
    report["summary"]["root_rot_format"] = root_rot_format
    return report


def write_report(report, json_path=None, csv_path=None):
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = report.get("frames", [])
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
