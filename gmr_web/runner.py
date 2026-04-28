import os
import pickle
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco as mj
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_BASE_DICT, ROBOT_XML_DICT, VIEWER_CAM_DISTANCE_DICT
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    get_gvhmr_data_offline_fast,
    load_smplx_file,
    load_gvhmr_pred_file,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.xsens import load_xsens_file

from gmr_web.common import ensure_dir


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
            offset = min(offset, float(pos[2]))
    return 0.0 if not np.isfinite(offset) else float(offset)


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
    smoothed[qpos_indices] = alpha * qpos[qpos_indices] + (1.0 - alpha) * prev_qpos[qpos_indices]
    return smoothed


def make_motion_data(qpos_list, fps):
    root_pos = np.asarray([qpos[:3] for qpos in qpos_list])
    root_rot = np.asarray([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.asarray([qpos[7:] for qpos in qpos_list])
    return {
        "fps": int(fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }


def save_motion_data(path, motion_data):
    with Path(path).open("wb") as f:
        pickle.dump(motion_data, f)


def load_motion_data(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def render_robot_motion(
    motion_path,
    video_path,
    *,
    robot="elf3",
    width=640,
    height=480,
    logger=None,
):
    motion_data = load_motion_data(motion_path)
    fps = int(motion_data["fps"])
    root_pos = np.asarray(motion_data["root_pos"])
    root_rot_wxyz = np.asarray(motion_data["root_rot"])[:, [3, 0, 1, 2]]
    dof_pos = np.asarray(motion_data["dof_pos"])

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=height, width=width)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance = VIEWER_CAM_DISTANCE_DICT[robot]
    camera.elevation = -10
    camera.azimuth = 90
    base_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ROBOT_BASE_DICT[robot])

    ensure_dir(Path(video_path).parent)
    writer = imageio.get_writer(
        str(video_path),
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
    )
    try:
        total = len(root_pos)
        for idx in range(total):
            data.qpos[:3] = root_pos[idx]
            data.qpos[3:7] = root_rot_wxyz[idx]
            data.qpos[7:] = dof_pos[idx]
            mj.mj_forward(model, data)
            if base_body_id >= 0:
                camera.lookat = data.xpos[base_body_id]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
            if logger and (idx == 0 or (idx + 1) % 100 == 0 or idx + 1 == total):
                logger(f"[Render] {idx + 1}/{total}")
    finally:
        writer.close()
        renderer.close()
    return Path(video_path)


def retarget_frames(
    motion_frames,
    *,
    src_human,
    robot,
    actual_human_height,
    fps,
    motion_path,
    ground_clearance=0.03,
    smoothing_alpha=1.0,
    apply_lower_body_smoothing=False,
    logger=None,
):
    def log(message):
        if logger:
            logger(message)

    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=robot,
        verbose=False,
    )
    ground_offset = estimate_ground_offset(retargeter, motion_frames) - ground_clearance
    retargeter.set_ground_offset(ground_offset)
    log(f"[GMR] Apply ground offset: {ground_offset:.4f} m")

    qpos_indices = get_smoothing_qpos_indices(retargeter.model) if apply_lower_body_smoothing else None
    prev_qpos = None
    qpos_list = []
    total = len(motion_frames)
    for idx, frame in enumerate(motion_frames):
        qpos = retargeter.retarget(frame)
        if apply_lower_body_smoothing:
            qpos = smooth_qpos(qpos, prev_qpos, qpos_indices, smoothing_alpha)
            prev_qpos = qpos.copy()
        qpos_list.append(qpos)
        if idx == 0 or (idx + 1) % 100 == 0 or idx + 1 == total:
            log(f"[GMR] Retarget {idx + 1}/{total}")

    save_motion_data(motion_path, make_motion_data(qpos_list, fps))
    return retargeter


def convert_gvhmr_pt(
    input_path,
    output_dir,
    *,
    robot="elf3",
    ground_clearance=0.03,
    generate_video=True,
    logger=None,
):
    output_dir = ensure_dir(output_dir)
    input_path = Path(input_path)
    motion_path = output_dir / "robot_motion.pkl"
    video_path = output_dir / "robot_preview.mp4"

    def log(message):
        if logger:
            logger(message)

    log("[GMR] Loading GVHMR SMPL-X motion")
    smplx_folder = Path(__file__).resolve().parents[1] / "assets" / "body_models"
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        str(input_path), smplx_folder
    )
    motion_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )

    log("[GMR] Retargeting GVHMR motion to ELF3")
    retarget_frames(
        motion_frames,
        src_human="smplx",
        robot=robot,
        actual_human_height=actual_human_height,
        fps=aligned_fps,
        motion_path=motion_path,
        ground_clearance=ground_clearance,
        logger=logger,
    )
    artifacts = {"motion_path": str(motion_path)}
    if generate_video:
        log("[GMR] Rendering preview video")
        render_robot_motion(motion_path, video_path, robot=robot, logger=logger)
        artifacts["video_path"] = str(video_path)
    return artifacts


def convert_smplx_npz(
    input_path,
    output_dir,
    *,
    robot="elf3",
    ground_clearance=0.03,
    generate_video=True,
    logger=None,
):
    output_dir = ensure_dir(output_dir)
    input_path = Path(input_path)
    motion_path = output_dir / "robot_motion.pkl"
    video_path = output_dir / "robot_preview.mp4"

    def log(message):
        if logger:
            logger(message)

    log("[GMR] Loading SMPL-X .npz motion")
    smplx_folder = Path(__file__).resolve().parents[1] / "assets" / "body_models"
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        str(input_path), smplx_folder
    )
    motion_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    log("[GMR] Retargeting SMPL-X motion to ELF3")
    retarget_frames(
        motion_frames,
        src_human="smplx",
        robot=robot,
        actual_human_height=actual_human_height,
        fps=aligned_fps,
        motion_path=motion_path,
        ground_clearance=ground_clearance,
        logger=logger,
    )
    artifacts = {"motion_path": str(motion_path)}
    if generate_video:
        log("[GMR] Rendering preview video")
        render_robot_motion(motion_path, video_path, robot=robot, logger=logger)
        artifacts["video_path"] = str(video_path)
    return artifacts


def convert_bvh(
    input_path,
    output_dir,
    *,
    source_type="bvh_xsens",
    robot="elf3",
    ground_clearance=0.03,
    smoothing_alpha=0.35,
    generate_video=True,
    logger=None,
):
    if source_type not in {"bvh_xsens", "bvh_lafan1", "bvh_nokov"}:
        raise ValueError(f"Unsupported BVH source type: {source_type}")

    output_dir = ensure_dir(output_dir)
    input_path = Path(input_path)
    motion_path = output_dir / "robot_motion.pkl"
    video_path = output_dir / "robot_preview.mp4"

    def log(message):
        if logger:
            logger(message)

    log(f"[GMR] Loading BVH motion ({source_type})")
    if source_type == "bvh_xsens":
        args = SimpleNamespace(
            bvh_file=str(input_path),
            scale=0.01,
            reset_to_zero=False,
            start=None,
            end=None,
            bvh_format="3DSM",
        )
        motion_frames, actual_human_height, frame_time = load_xsens_file(args)
        motion_fps = max(1, round(1 / frame_time))
        apply_smoothing = True
    else:
        bvh_format = "lafan1" if source_type == "bvh_lafan1" else "nokov"
        motion_frames, actual_human_height = load_bvh_file(str(input_path), format=bvh_format)
        motion_fps = 30
        apply_smoothing = False

    log(f"[GMR] Retargeting {source_type} motion to ELF3")
    retarget_frames(
        motion_frames,
        src_human=source_type,
        robot=robot,
        actual_human_height=actual_human_height,
        fps=motion_fps,
        motion_path=motion_path,
        ground_clearance=ground_clearance,
        smoothing_alpha=smoothing_alpha,
        apply_lower_body_smoothing=apply_smoothing,
        logger=logger,
    )
    artifacts = {"motion_path": str(motion_path)}
    if generate_video:
        log("[GMR] Rendering preview video")
        render_robot_motion(motion_path, video_path, robot=robot, logger=logger)
        artifacts["video_path"] = str(video_path)
    return artifacts


def convert_fbx_offline(
    input_path,
    output_dir,
    *,
    robot="elf3",
    ground_clearance=0.03,
    generate_video=True,
    logger=None,
):
    output_dir = ensure_dir(output_dir)
    input_path = Path(input_path)
    motion_path = output_dir / "robot_motion.pkl"
    video_path = output_dir / "robot_preview.mp4"

    def log(message):
        if logger:
            logger(message)

    log("[GMR] Loading fbx_offline motion frames")
    with input_path.open("rb") as f:
        motion_frames = pickle.load(f)
    if not isinstance(motion_frames, (list, tuple)) or not motion_frames:
        raise ValueError("fbx_offline .pkl must contain a non-empty list of GMR human frames.")

    log("[GMR] Retargeting fbx_offline motion to ELF3")
    retarget_frames(
        motion_frames,
        src_human="fbx_offline",
        robot=robot,
        actual_human_height=1.8,
        fps=120,
        motion_path=motion_path,
        ground_clearance=ground_clearance,
        logger=logger,
    )
    artifacts = {"motion_path": str(motion_path)}
    if generate_video:
        log("[GMR] Rendering preview video")
        render_robot_motion(motion_path, video_path, robot=robot, logger=logger)
        artifacts["video_path"] = str(video_path)
    return artifacts
