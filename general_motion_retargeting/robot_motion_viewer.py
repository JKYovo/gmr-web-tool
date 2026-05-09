import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1


def add_marker(viewer, pos, size, rgba, label=None):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=size,
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    if label is not None:
        geom.label = label
    viewer.user_scn.ngeom += 1


def add_line(viewer, start, end, width, rgba):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=[width, 0.0, 0.0],
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=start,
        to=end,
    )
    viewer.user_scn.ngeom += 1


def add_box(viewer, pos, mat, size, rgba, label=None):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
        mat=mat.flatten(),
        rgba=rgba,
    )
    if label is not None:
        geom.label = label
    viewer.user_scn.ngeom += 1


def convex_hull_xy(points):
    points = sorted(set((float(point[0]), float(point[1])) for point in points))
    if len(points) <= 1:
        return np.asarray(points)

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


def ground_yaw_rotation(body_mat):
    x_axis = body_mat[:, 0].copy()
    x_axis[2] = 0.0
    norm = np.linalg.norm(x_axis)
    if norm < 1e-6:
        return np.eye(3)
    x_axis /= norm
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def support_foot_corners(center, yaw_mat, half_length, half_width, z=0.003):
    local_corners = np.array(
        [
            [half_length, half_width, 0.0],
            [half_length, -half_width, 0.0],
            [-half_length, -half_width, 0.0],
            [-half_length, half_width, 0.0],
        ]
    )
    corners = center + local_corners @ yaw_mat.T
    corners[:, 2] = z
    return corners


class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480,
                keyboard_callback=None,
                ):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.base_geom_rgba = self.model.geom_rgba.copy()
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video


        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False, 
            key_callback=keyboard_callback
            )      

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        self.base_transparent_flag = transparent_robot
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(
                self.video_path,
                fps=self.motion_fps,
                format="FFMPEG",
                codec="libx264",
                pixelformat="yuv420p",
            )
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=None,
            camera_mode="fixed",
            show_com_projection=False,
            show_support_polygon=False,
            show_self_collision=False,
            collision_visual_mode="opaque",
            collision_penetration_epsilon=1e-4,
            collision_robot_alpha=0.35,
            show_collision_labels=False,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos
        
        mj.mj_forward(self.model, self.data)
        collision_contacts = []
        if show_self_collision:
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = (
                1 if collision_visual_mode == "transparent" else self.base_transparent_flag
            )
            collision_contacts = self._collect_self_collision_contacts(
                penetration_epsilon=collision_penetration_epsilon
            )
            self._apply_collision_geom_colors(
                collision_contacts,
                visual_mode=collision_visual_mode,
                robot_alpha=collision_robot_alpha,
            )
        else:
            self.viewer.opt.flags[
                mj.mjtVisFlag.mjVIS_TRANSPARENT
            ] = self.base_transparent_flag
            self._restore_geom_colors()

        if follow_camera is not None:
            camera_mode = "fixed" if follow_camera else "free"
        
        if camera_mode in ("fixed", "track"):
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
        if camera_mode == "fixed":
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 正面视角，轻微向下看
            # self.viewer.cam.azimuth = 180    # 正面朝向机器人
        
        # Clean custom geometry.
        self.viewer.user_scn.ngeom = 0

        if show_com_projection:
            total_mass = np.sum(self.model.body_mass)
            if total_mass > 0.0:
                com = np.sum(self.data.xipos * self.model.body_mass[:, None], axis=0) / total_mass
                com_projection = com.copy()
                com_projection[2] = 0.0
                add_line(
                    self.viewer,
                    com_projection,
                    com,
                    width=0.006,
                    rgba=np.array([1.0, 0.85, 0.0, 0.75]),
                )
                add_marker(
                    self.viewer,
                    com,
                    size=np.array([0.025, 0.025, 0.025]),
                    rgba=np.array([1.0, 0.1, 0.1, 1.0]),
                )
                add_marker(
                    self.viewer,
                    com_projection,
                    size=np.array([0.035, 0.035, 0.006]),
                    rgba=np.array([1.0, 0.85, 0.0, 1.0]),
                )

        if show_support_polygon:
            support_corners = []
            foot_specs = (
                ("l_ankle_x_link", "lf_tc"),
                ("r_ankle_x_link", "rf_tc"),
            )
            for foot_body_name, foot_site_name in foot_specs:
                body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, foot_body_name)
                site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, foot_site_name)
                if body_id < 0 or site_id < 0:
                    continue
                site_pos = self.data.site_xpos[site_id]
                if site_pos[2] > 0.08:
                    continue
                yaw_mat = ground_yaw_rotation(self.data.xmat[body_id].reshape(3, 3))
                center = self.data.xpos[body_id].copy()
                center[2] = 0.004
                half_length = 0.11
                half_width = 0.04
                add_box(
                    self.viewer,
                    center,
                    yaw_mat,
                    size=np.array([half_length, half_width, 0.003]),
                    rgba=np.array([0.0, 0.55, 1.0, 0.22]),
                )
                corners = support_foot_corners(center, yaw_mat, half_length, half_width)
                support_corners.extend(corners)
                for start, end in zip(corners, np.roll(corners, -1, axis=0)):
                    add_line(
                        self.viewer,
                        start,
                        end,
                        width=0.004,
                        rgba=np.array([0.0, 0.65, 1.0, 0.9]),
                    )

            if len(support_corners) >= 6:
                hull = convex_hull_xy(support_corners)
                hull3 = np.column_stack(
                    [hull[:, 0], hull[:, 1], np.full(len(hull), 0.01)]
                )
                for start, end in zip(hull3, np.roll(hull3, -1, axis=0)):
                    add_line(
                        self.viewer,
                        start,
                        end,
                        width=0.007,
                        rgba=np.array([0.0, 1.0, 0.35, 1.0]),
                    )

        if show_self_collision:
            self._draw_collision_contacts(
                collision_contacts,
                show_labels=show_collision_labels,
            )

        if human_motion_data is not None:
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None
                    )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)

    def _restore_geom_colors(self):
        self.model.geom_rgba[:] = self.base_geom_rgba

    def _apply_collision_geom_colors(self, collision_contacts, visual_mode, robot_alpha):
        rgba = self.base_geom_rgba.copy()
        if visual_mode == "transparent" and robot_alpha is not None:
            alpha = float(np.clip(robot_alpha, 0.05, 1.0))
            for geom_id in range(self.model.ngeom):
                geom_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom_id)
                if geom_name != "floor":
                    rgba[geom_id, 3] = min(rgba[geom_id, 3], alpha)

        for contact in collision_contacts:
            rgba[contact["geom1"], :] = np.array([1.0, 0.03, 0.0, 1.0])
            rgba[contact["geom2"], :] = np.array([1.0, 0.45, 0.0, 1.0])

        self.model.geom_rgba[:] = rgba

    def _collect_self_collision_contacts(self, penetration_epsilon=1e-4):
        contacts = []
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            if contact.dist >= -penetration_epsilon:
                continue

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            geom1_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom1) or ""
            geom2_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom2) or ""
            if not self._is_robot_collision_geom(geom1_name):
                continue
            if not self._is_robot_collision_geom(geom2_name):
                continue

            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if body1 == body2:
                continue

            contacts.append(
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "geom1_name": geom1_name,
                    "geom2_name": geom2_name,
                    "body1_name": mj.mj_id2name(
                        self.model, mj.mjtObj.mjOBJ_BODY, body1
                    )
                    or "",
                    "body2_name": mj.mj_id2name(
                        self.model, mj.mjtObj.mjOBJ_BODY, body2
                    )
                    or "",
                    "pos": contact.pos.copy(),
                    "dist": float(contact.dist),
                }
            )
        return contacts

    @staticmethod
    def _is_robot_collision_geom(geom_name):
        return bool(geom_name) and geom_name != "floor" and "collision" in geom_name

    def _draw_collision_contacts(self, collision_contacts, show_labels=False):
        for contact in collision_contacts:
            penetration = max(0.0, -contact["dist"])
            radius = float(np.clip(0.025 + 0.8 * penetration, 0.025, 0.08))
            label = None
            if show_labels:
                label = (
                    f"{contact['body1_name']} <-> {contact['body2_name']} "
                    f"{100.0 * penetration:.1f}cm"
                )
            add_marker(
                self.viewer,
                contact["pos"],
                size=np.array([radius, radius, radius]),
                rgba=np.array([0.72, 0.0, 1.0, 0.92]),
                label=label,
            )

            geom1_pos = self.data.geom_xpos[contact["geom1"]]
            geom2_pos = self.data.geom_xpos[contact["geom2"]]
            add_line(
                self.viewer,
                geom1_pos,
                geom2_pos,
                width=0.01,
                rgba=np.array([0.72, 0.0, 1.0, 0.7]),
            )
    
    def close(self):
        self._restore_geom_colors()
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
