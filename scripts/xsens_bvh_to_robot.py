import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.xsens import load_xsens_file
import mujoco as mj
from rich import print
from tqdm import tqdm
import os
import numpy as np


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


def get_smoothing_qpos_indices(model):
    indices = [0, 1, 2]  # root position
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


def build_retarget_options(args, motion_fps):
    enable_all = args.enable_robot_constraints
    return {
        "velocity_limits": {
            "enabled": enable_all or args.enable_velocity_limit,
        },
        "collision_avoidance": {
            "enabled": enable_all or args.enable_collision_avoidance,
        },
        "support_foot": {
            "enabled": enable_all or args.enable_support_foot,
            "motion_fps": motion_fps,
        },
        "stability": {
            "enabled": enable_all or args.enable_stability_weighting,
        },
    }


if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        # default="./xsens_bvh/xsens_walk.bvh",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_h1_2",
            "Q1",
            "X1",
            "elf3",
        ],
        default="unitree_h1_2",
    )

    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--no_viewer",
        action="store_true",
        default=False,
        help="Run retargeting headlessly. Save pkl without opening the MuJoCo viewer.",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--save_path",
        # default="./retargeting_data/unitree_h1_2_xsens_walk.pkl",
        default=None,
        help="Path to save the robot motion.",
    )

    # parser.add_argument(
    #     "--axis_order",
    #     default="zxy",
    #     help="",
    # )

    parser.add_argument(
        "--scale",
        default=0.01,
        type=float,
        help="The scaling size is determined based on the units used for displacement",
    )

    parser.add_argument(
        "--reset_to_zero",
        action="store_true",
        default=False,
        help="Set the displacement and Z-axis rotation to zero",
    )

    parser.add_argument(
        "--start",
        default=None,
        type=int,
        help="The sequence number of the first frame that you want to process",
    )

    parser.add_argument(
        "--end",
        default=None,
        type=int,
        help="The sequence number of the last frame that you want to process",
    )

    parser.add_argument(
        "--bvh_format",
        default="3DSM",
        type=str,
        choices=[
            "3DSM",
        ],
        help="The format of bvh files,3ds Max,MotionBuilder,and P6?",
    )

    parser.add_argument(
        "--ground_clearance",
        type=float,
        default=0.03,
        help="Extra lift in meters applied after ground alignment to avoid initial foot penetration.",
    )

    parser.add_argument(
        "--smoothing_alpha",
        type=float,
        default=0.35,
        help="EMA factor for root/lower-body temporal smoothing. Set to 1.0 to disable.",
    )

    parser.add_argument(
        "--show_com_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show robot center of mass and its ground projection in the MuJoCo viewer.",
    )

    parser.add_argument(
        "--show_support_polygon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show estimated support foot area/polygon in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--show_self_collision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Highlight current robot self-collision geoms and contact points.",
    )
    parser.add_argument(
        "--collision_visual_mode",
        choices=["opaque", "transparent"],
        default="opaque",
        help=(
            "Self-collision display mode. opaque keeps normal depth cues and only "
            "colors colliding parts; transparent makes the whole robot see-through."
        ),
    )
    parser.add_argument(
        "--collision_robot_alpha",
        type=float,
        default=0.35,
        help="Robot geom alpha when --collision_visual_mode transparent is enabled.",
    )
    parser.add_argument(
        "--show_collision_labels",
        action="store_true",
        default=False,
        help="Show collision body labels. Off by default to avoid blocking the view.",
    )

    parser.add_argument(
        "--free_camera",
        action="store_true",
        default=False,
        help="Shortcut for --camera_mode free.",
    )

    parser.add_argument(
        "--camera_mode",
        choices=["track", "fixed", "free"],
        default="track",
        help=(
            "Viewer camera behavior: track keeps the look-at point on the robot "
            "but lets the mouse control rotation/zoom; fixed also resets distance "
            "and elevation every frame; free never updates the camera."
        ),
    )

    parser.add_argument(
        "--enable_robot_constraints",
        action="store_true",
        default=False,
        help=(
            "Enable the experimental source-level robot feasibility constraints "
            "(velocity, collision, support foot, and stability weighting)."
        ),
    )

    parser.add_argument(
        "--enable_velocity_limit",
        action="store_true",
        default=False,
        help="Enable grouped joint velocity limits during IK.",
    )

    parser.add_argument(
        "--enable_collision_avoidance",
        action="store_true",
        default=False,
        help="Enable configured self-collision avoidance during IK.",
    )

    parser.add_argument(
        "--enable_support_foot",
        action="store_true",
        default=False,
        help="Enable dynamic support-foot task weighting during IK.",
    )

    parser.add_argument(
        "--enable_stability_weighting",
        action="store_true",
        default=False,
        help="Enable COM/support-margin based task weighting during IK.",
    )

    args = parser.parse_args()

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    # Load SMPLX trajectory
    # lafan1_data_frames, actual_human_height = load_lafan1_file(args.bvh_file)
    lafan1_data_frames, actual_human_height, frame_time = load_xsens_file(args)

    # Initialize the retargeting system
    motion_fps = max(1, round(1 / frame_time))
    retargeter = GMR(
        src_human="bvh_xsens",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        retarget_options=build_retarget_options(args, motion_fps),
    )
    ground_offset = estimate_ground_offset(retargeter, lafan1_data_frames) - args.ground_clearance
    retargeter.set_ground_offset(ground_offset)
    print(f"Apply ground offset: {ground_offset:.4f} m")
    qpos_smoothing_indices = get_smoothing_qpos_indices(retargeter.model)
    prev_smoothed_qpos = None

    robot_motion_viewer = None
    if not args.no_viewer:
        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=motion_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path,
            # video_width=2080,
            # video_height=1170
        )

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds

    print(f"mocap_frame_rate: {motion_fps}")

    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")

    # Start the viewer
    i = 0

    while i < len(lafan1_data_frames):

        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            # print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)
        qpos = smooth_qpos(
            qpos, prev_smoothed_qpos, qpos_smoothing_indices, args.smoothing_alpha
        )
        prev_smoothed_qpos = qpos.copy()

        # visualize
        if robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                camera_mode="free" if args.free_camera else args.camera_mode,
                show_com_projection=args.show_com_projection,
                show_support_polygon=args.show_support_polygon,
                show_self_collision=args.show_self_collision,
                collision_visual_mode=args.collision_visual_mode,
                collision_robot_alpha=args.collision_robot_alpha,
                show_collision_labels=args.show_collision_labels,
                # human_pos_offset=np.array([0.0, 0.0, 0.0])
            )

        i += 1

        if args.save_path is not None:
            qpos_list.append(qpos)

    if args.save_path is not None:
        import pickle

        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # MuJoCo qpos stores root quaternions as wxyz, while GMR motion pkl
        # files use xyzw. Keep the saved format consistent with gvhmr_to_robot
        # and data_loader.load_robot_motion().
        root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None

        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()

    if robot_motion_viewer is not None:
        robot_motion_viewer.close()
