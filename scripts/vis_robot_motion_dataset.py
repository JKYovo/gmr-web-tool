from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm

paused = False
motion_num = 0
motion_id = 0
current_motion_id = -1

def keyboard_callback(keycode):
    global paused, motion_id, motion_num
    if chr(keycode) == ' ':
        paused = not paused
    if chr(keycode) == '[':
        motion_id = (motion_id - 1) % motion_num
    if chr(keycode) == ']':
        motion_id = (motion_id + 1) % motion_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_folder", type=str, required=True)

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
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_folder = args.robot_motion_folder
    camera_mode = "free" if args.free_camera else args.camera_mode
    
    if not os.path.exists(robot_motion_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_folder} does not exist.")
    
    motion_files = [f for f in os.listdir(robot_motion_folder) if f.endswith('.pkl')]
    motion_files = sorted(motion_files)
    motion_num = len(motion_files)
    print(f"Found {motion_num} motion files in {robot_motion_folder}, loading...")
    motion_dataset = []
    for motion_file in tqdm(motion_files):
        motion_path = os.path.join(robot_motion_folder, motion_file)
        motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(motion_path)
        motion_dataset.append({
            "motion_file": motion_file,
            "motion_data": motion_data,
            "motion_fps": motion_fps,
            "motion_root_pos": motion_root_pos,
            "motion_root_rot": motion_root_rot,
            "motion_dof_pos": motion_dof_pos,
            "motion_local_body_pos": motion_local_body_pos,
            "motion_link_body_list": motion_link_body_list,
        })
    print("Loading done.")
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path, 
                            keyboard_callback=keyboard_callback)
    
    frame_idx = 0
    while True:
        # get current motion
        if current_motion_id != motion_id:
            current_motion_id = motion_id
            frame_idx = 0
            motion_data = motion_dataset[motion_id]
            motion_file = motion_data["motion_file"]
            motion_fps = motion_data["motion_fps"]
            motion_root_pos = motion_data["motion_root_pos"]
            motion_root_rot = motion_data["motion_root_rot"]
            motion_dof_pos = motion_data["motion_dof_pos"]
            print(f"Switched to motion {motion_id}: {motion_file}, fps: {motion_fps}, num_frames: {len(motion_root_pos)}")
        
        
        if not paused:
            env.step(motion_root_pos[frame_idx], 
                    motion_root_rot[frame_idx], 
                    motion_dof_pos[frame_idx], 
                    rate_limit=True,
                    camera_mode=camera_mode,
                    show_com_projection=args.show_com_projection,
                    show_support_polygon=args.show_support_polygon,
                    show_self_collision=args.show_self_collision,
                    collision_visual_mode=args.collision_visual_mode,
                    collision_robot_alpha=args.collision_robot_alpha,
                    show_collision_labels=args.show_collision_labels)
            frame_idx += 1
            if frame_idx >= len(motion_root_pos):
                frame_idx = 0
    env.close()
