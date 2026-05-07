from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
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
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    camera_mode = "free" if args.free_camera else args.camera_mode
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        env.step(motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx], 
                rate_limit=True,
                camera_mode=camera_mode,
                show_com_projection=args.show_com_projection,
                show_support_polygon=args.show_support_polygon)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()
