import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print


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


def estimate_ground_offset(retargeter: GMR, motion_frames):
    # 估计人体动作里最低的关键点高度。
    #
    # 为什么要做这个？
    # GVHMR 输出的人体动作不一定刚好站在 MuJoCo 地面 z=0 上。
    # 如果直接 retarget，机器人一开始可能脚插进地面，或者整个人悬空。
    # 这里先把所有人体帧扫一遍，找出最低点，再用它来设置 ground_offset。
    offset = np.inf
    for human_data in motion_frames:
        # GMR 内部会把输入统一转成 numpy，并按真实人体身高做尺度缩放。
        # 这样不同身高的人体动作可以映射到机器人尺寸上。
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(
            human_data, retargeter.human_root_name, retargeter.human_scale_table
        )
        # 应用 JSON 配置里的初始位置/旋转偏移。
        # 这些偏移用于把 SMPLX 的人体坐标系对齐到 GMR 期望的坐标系。
        human_data = retargeter.offset_human_data(
            human_data, retargeter.pos_offsets1, retargeter.rot_offsets1
        )
        # human_data 里每个 value 基本是一个人体关键点/关节的 (位置, 旋转)。
        # 这里只关心位置 pos 的 z 值，也就是高度。
        for pos, _quat in human_data.values():
            if pos[2] < offset:
                offset = pos[2]
    if not np.isfinite(offset):
        return 0.0
    return float(offset)


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    # 这个脚本的主输入是 GVHMR 生成的 hmr4d_results.pt。
    # 输出可以是：
    # 1. MuJoCo 预览窗口/录制视频
    # 2. robot_motion.pkl，里面保存机器人 root 和各关节角度序列
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_pred_file",
        help="SMPLX motion file to load.",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "elf3"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--ground_clearance",
        type=float,
        default=0.03,
        help="Extra lift in meters applied after ground alignment to avoid initial foot penetration.",
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


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # 第一步：读取 GVHMR 的人体动作结果。
    #
    # load_gvhmr_pred_file 会把 hmr4d_results.pt 转成 GMR 能用的 SMPLX 数据。
    # 这里的 actual_human_height 很重要：GMR 会用它估计人体到机器人之间的缩放比例。
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )
    
    # 第二步：统一帧率。
    #
    # GVHMR 的输出可能来自不同 fps 的视频。GMR retarget 时这里固定整理成 30fps，
    # 得到 smplx_data_frames，也就是“逐帧人体动作数据”。
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
    
   
    # 第三步：创建 GMR retarget 对象。
    #
    # src_human="smplx" 表示输入人体格式是 SMPLX。
    # tgt_robot=args.robot 表示输出目标机器人，比如 elf3 / unitree_g1。
    #
    # 真正的“人体点 -> 机器人点”绑定关系，不在这个脚本里硬编码，
    # 而是在 general_motion_retargeting/ik_configs/*_to_*.json 里定义。
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        retarget_options=build_retarget_options(args, aligned_fps),
    )
    # 第四步：把动作整体对齐到地面。
    #
    # estimate_ground_offset 会找人体动作最低点；
    # ground_clearance 是额外抬高一点点，避免机器人初始脚底穿地。
    ground_offset = estimate_ground_offset(retarget, smplx_data_frames) - args.ground_clearance
    retarget.set_ground_offset(ground_offset)
    print(f"Apply ground offset: {ground_offset:.4f} m")
    
    # 第五步：创建 MuJoCo 预览器。
    #
    # record_video=True 时，它会把后面 step() 播放过的机器人动作录成 mp4。
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.gvhmr_pred_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        # 如果传了 --save_path，就把每一帧解算出来的 qpos 存起来，
        # 最后统一写成 robot_motion.pkl。
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # 第六步：逐帧 retarget。
    #
    # 这是脚本最核心的循环：
    # 1. 取一帧 SMPLX 人体动作
    # 2. 调用 retarget.retarget(...) 解算机器人姿态
    # 3. 把机器人姿态送进 MuJoCo 预览器
    # 4. 如果需要保存，就把这一帧 qpos 收集起来
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # 当前这一帧的人体动作目标。
        # 可以把它理解成“这一帧人体的手、肘、膝、脚等关键点在哪里”。
        smplx_data = smplx_data_frames[i]

        # 核心解算：人体动作 -> 机器人 qpos。
        #
        # qpos 是 MuJoCo 里机器人的完整姿态向量，一般结构是：
        # qpos[:3]   = root_pos，机器人根节点位置
        # qpos[3:7]  = root_rot，机器人根节点四元数旋转，GMR/MuJoCo 这里是 wxyz
        # qpos[7:]   = dof_pos，机器人各个关节角
        qpos = retarget.retarget(smplx_data)

        # 把当前帧机器人姿态送进预览器。
        #
        # 如果 record_video=True，viewer 内部也会把这些帧录下来。
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            show_self_collision=args.show_self_collision,
            collision_visual_mode=args.collision_visual_mode,
            collision_robot_alpha=args.collision_robot_alpha,
            show_collision_labels=args.show_collision_labels,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        # 第七步：把逐帧 qpos 整理成 GMR 标准 robot_motion.pkl。
        #
        # 这个 pkl 是后面 Web、后处理脚本、渲染预览都会读取的核心数据。
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # GMR retarget 得到的 root_rot 是 wxyz，
        # 但保存到 robot_motion.pkl 时转成 xyzw。
        # 这就是为什么这里用了 [1, 2, 3, 0] 重排。
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    # 关闭 MuJoCo viewer，释放窗口和录制资源。
    robot_motion_viewer.close()
