
import fnmatch
import json

import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
        retarget_options: dict | None=None,
    ) -> None:
        self.verbose = verbose

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        self.robot_geom_names = {}
        for i in range(self.model.ngeom):
            geom_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i)
            if geom_name:
                self.robot_geom_names[geom_name] = i
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        self.ik_config = ik_config
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.task_infos1 = []
        self.task_infos2 = []
        self.all_task_infos = []
        self.frame_to_task_infos = {}

        self.prev_human_foot_pos = {}
        self.last_support_state = {}
        self.last_stability_metrics = {}
        self.last_ik_velocity_stats = {}
        self.ik_velocity_stats_history = []
        self.collision_solve_failures = 0
        self.collision_avoidance_limit = None
        self.retarget_options = self._normalize_retarget_options(
            ik_config, retarget_options, use_velocity_limit
        )
        self.ik_limits = self._build_ik_limits()
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    @staticmethod
    def _deep_update(base, updates):
        merged = dict(base)
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = GeneralMotionRetargeting._deep_update(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _normalize_retarget_options(self, ik_config, retarget_options, use_velocity_limit):
        options = {
            "velocity_limits": {
                "enabled": False,
                "default": 3.0 * np.pi,
                "waist": 4.0,
                "leg": 5.0,
                "arm": 5.5,
                "wrist": 7.0,
                "joint_limits": {},
                "max_extra_iterations": None,
            },
            "collision_avoidance": {
                "enabled": False,
                "gain": 0.85,
                "min_distance": 0.02,
                "detection_distance": 0.08,
                "bound_relaxation": 0.0,
                "fail_open": True,
                "pairs": [],
            },
            "support_foot": {
                "enabled": False,
                "left_robot_frame": "l_ankle_x_link",
                "right_robot_frame": "r_ankle_x_link",
                "height_threshold": 0.06,
                "velocity_threshold": 0.25,
                "motion_fps": None,
                "support_position_multiplier": 1.5,
                "support_orientation_multiplier": 1.25,
                "swing_position_multiplier": 0.85,
                "swing_orientation_multiplier": 0.85,
            },
            "stability": {
                "enabled": False,
                "support_height": 0.08,
                "foot_length": 0.20,
                "foot_width": 0.11,
                "margin_threshold": 0.02,
                "critical_margin": -0.02,
                "torso_multiplier": 1.25,
                "waist_multiplier": 1.2,
                "arm_multiplier": 0.85,
                "left_foot_body": "l_ankle_x_link",
                "right_foot_body": "r_ankle_x_link",
            },
        }

        for section in options.keys():
            if section in ik_config:
                options[section] = self._deep_update(options[section], ik_config[section])

        if use_velocity_limit:
            options["velocity_limits"]["enabled"] = True

        aliases = {
            "velocity_limit": "velocity_limits",
            "velocity_limits": "velocity_limits",
            "collision_avoidance": "collision_avoidance",
            "support_foot": "support_foot",
            "support_foot_lock": "support_foot",
            "stability": "stability",
            "stability_weighting": "stability",
        }
        if retarget_options:
            for key, value in retarget_options.items():
                canonical_key = aliases.get(key)
                if canonical_key is None:
                    continue
                if isinstance(value, bool):
                    options[canonical_key]["enabled"] = value
                elif isinstance(value, dict):
                    options[canonical_key] = self._deep_update(
                        options[canonical_key], value
                    )
                elif value is not None:
                    options[canonical_key]["enabled"] = bool(value)

        return options

    def _build_ik_limits(self):
        limits = [mink.ConfigurationLimit(self.model)]

        velocity_limits = self._build_velocity_limits()
        if velocity_limits:
            limits.append(mink.VelocityLimit(self.model, velocity_limits))
            if self.verbose:
                print(
                    f"[GMR] Velocity limits enabled for {len(velocity_limits)} joints."
                )

        collision_config = self.retarget_options["collision_avoidance"]
        if collision_config.get("enabled", False):
            collision_pairs = self._build_collision_pairs(collision_config)
            if collision_pairs:
                self.collision_avoidance_limit = mink.CollisionAvoidanceLimit(
                    self.model,
                    collision_pairs,
                    gain=float(collision_config.get("gain", 0.85)),
                    minimum_distance_from_collisions=float(
                        collision_config.get("min_distance", 0.02)
                    ),
                    collision_detection_distance=float(
                        collision_config.get("detection_distance", 0.08)
                    ),
                    bound_relaxation=float(collision_config.get("bound_relaxation", 0.0)),
                )
                limits.append(self.collision_avoidance_limit)
                if self.verbose:
                    print(
                        f"[GMR] Collision avoidance enabled for "
                        f"{len(collision_pairs)} configured geom groups."
                    )
            elif self.verbose:
                print("[GMR] Collision avoidance enabled, but no valid geom pairs found.")

        return limits

    def _build_velocity_limits(self):
        config = self.retarget_options["velocity_limits"]
        if not config.get("enabled", False):
            return {}

        default_limit = float(config.get("default", 3.0 * np.pi))
        joint_limits = config.get("joint_limits", {})
        velocity_limits = {}

        for joint_name in self.robot_dof_names.keys():
            if not joint_name:
                continue
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            if self.model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
                continue

            if joint_name in joint_limits:
                max_velocity = float(joint_limits[joint_name])
            elif "waist" in joint_name:
                max_velocity = float(config.get("waist", default_limit))
            elif any(token in joint_name for token in ("hip", "knee", "ankle")):
                max_velocity = float(config.get("leg", default_limit))
            elif "wrist" in joint_name:
                max_velocity = float(config.get("wrist", default_limit))
            elif any(token in joint_name for token in ("shoulder", "elbow")):
                max_velocity = float(config.get("arm", default_limit))
            else:
                max_velocity = default_limit

            velocity_limits[joint_name] = max_velocity

        return velocity_limits

    def _build_collision_pairs(self, collision_config):
        raw_pairs = (
            collision_config.get("pairs")
            or collision_config.get("self_collision_pairs")
            or []
        )
        collision_pairs = []
        for raw_pair in raw_pairs:
            if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                continue
            left_group = self._expand_collision_group(raw_pair[0])
            right_group = self._expand_collision_group(raw_pair[1])
            if left_group and right_group:
                collision_pairs.append((left_group, right_group))
            elif self.verbose:
                print(
                    f"[GMR] Skip collision pair {raw_pair}: "
                    f"expanded to {left_group} / {right_group}."
                )
        return collision_pairs

    def _expand_collision_group(self, names):
        if isinstance(names, str):
            names = [names]
        expanded = []
        for name in names:
            if not name:
                continue
            if "*" in name:
                for geom_name in self.robot_geom_names.keys():
                    if fnmatch.fnmatch(geom_name, name) and self._is_collision_geom(geom_name):
                        expanded.append(geom_name)
                for body_name in self.robot_body_names.keys():
                    if fnmatch.fnmatch(body_name, name):
                        expanded.extend(self._collision_geoms_for_body(body_name))
                continue

            if name in self.robot_geom_names:
                expanded.append(name)
                continue

            if name in self.robot_body_names:
                expanded.extend(self._collision_geoms_for_body(name))
                continue

            candidate = name.replace("_link", "_collision")
            if candidate in self.robot_geom_names:
                expanded.append(candidate)

        seen = set()
        unique = []
        for geom_name in expanded:
            if geom_name in seen:
                continue
            seen.add(geom_name)
            unique.append(geom_name)
        return unique

    def _collision_geoms_for_body(self, body_name):
        body_id = self.robot_body_names.get(body_name)
        if body_id is None:
            return []
        geoms = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != int(body_id):
                continue
            geom_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, geom_id)
            if self._is_collision_geom(geom_name):
                geoms.append(geom_name)
        return geoms

    @staticmethod
    def _is_collision_geom(geom_name):
        return bool(geom_name) and geom_name != "floor" and "collision" in geom_name

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
                self._register_task_info(
                    table_id=1,
                    frame_name=frame_name,
                    human_body_name=body_name,
                    task=task,
                )
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []
                self._register_task_info(
                    table_id=2,
                    frame_name=frame_name,
                    human_body_name=body_name,
                    task=task,
                )

    def _register_task_info(self, table_id, frame_name, human_body_name, task):
        info = {
            "table_id": table_id,
            "frame_name": frame_name,
            "human_body_name": human_body_name,
            "task": task,
            "base_position_cost": task.cost[:3].copy(),
            "base_orientation_cost": task.cost[3:].copy(),
            "current_position_cost": task.cost[:3].copy(),
            "current_orientation_cost": task.cost[3:].copy(),
        }
        if table_id == 1:
            self.task_infos1.append(info)
        else:
            self.task_infos2.append(info)
        self.all_task_infos.append(info)
        self.frame_to_task_infos.setdefault(frame_name, []).append(info)

    def _set_task_cost(self, info, position_cost, orientation_cost):
        info["task"].set_position_cost(position_cost)
        info["task"].set_orientation_cost(orientation_cost)
        info["current_position_cost"] = np.asarray(position_cost, dtype=float).copy()
        info["current_orientation_cost"] = np.asarray(orientation_cost, dtype=float).copy()

    def _reset_dynamic_task_costs(self):
        for info in self.all_task_infos:
            self._set_task_cost(
                info, info["base_position_cost"], info["base_orientation_cost"]
            )

    def _scale_task_cost(self, info, position_multiplier=1.0, orientation_multiplier=1.0):
        position_cost = info["current_position_cost"] * float(position_multiplier)
        orientation_cost = info["current_orientation_cost"] * float(orientation_multiplier)
        self._set_task_cost(info, position_cost, orientation_cost)

    def _apply_dynamic_task_weights(self):
        self._reset_dynamic_task_costs()

        if self.retarget_options["support_foot"].get("enabled", False):
            self._apply_support_foot_weighting()

        if self.retarget_options["stability"].get("enabled", False):
            self._apply_stability_weighting()

    def _apply_support_foot_weighting(self):
        config = self.retarget_options["support_foot"]
        left_frame = config.get("left_robot_frame", "l_ankle_x_link")
        right_frame = config.get("right_robot_frame", "r_ankle_x_link")
        foot_frames = {"left": left_frame, "right": right_frame}

        foot_positions = {}
        foot_heights = {}
        for side, frame_name in foot_frames.items():
            task_infos = self.frame_to_task_infos.get(frame_name, [])
            if not task_infos:
                continue
            human_body_name = task_infos[0]["human_body_name"]
            if human_body_name not in self.scaled_human_data:
                continue
            foot_pos = np.asarray(self.scaled_human_data[human_body_name][0], dtype=float)
            foot_positions[side] = foot_pos
            foot_heights[side] = float(foot_pos[2])

        if not foot_positions:
            self.last_support_state = {}
            return

        min_height = min(foot_heights.values())
        height_threshold = float(config.get("height_threshold", 0.06))
        velocity_threshold = float(config.get("velocity_threshold", 0.25))
        motion_fps = config.get("motion_fps")
        motion_fps = float(motion_fps) if motion_fps else None

        support_state = {}
        for side, foot_pos in foot_positions.items():
            prev_pos = self.prev_human_foot_pos.get(side)
            if prev_pos is None:
                foot_speed = 0.0
            else:
                displacement = float(np.linalg.norm(foot_pos - prev_pos))
                foot_speed = displacement * motion_fps if motion_fps else displacement
            near_ground = foot_heights[side] <= min_height + height_threshold
            is_support = near_ground and foot_speed <= velocity_threshold
            support_state[side] = {
                "support": bool(is_support),
                "height": foot_heights[side],
                "speed": foot_speed,
            }

            frame_name = foot_frames[side]
            for info in self.frame_to_task_infos.get(frame_name, []):
                if is_support:
                    self._scale_task_cost(
                        info,
                        config.get("support_position_multiplier", 1.5),
                        config.get("support_orientation_multiplier", 1.25),
                    )
                else:
                    self._scale_task_cost(
                        info,
                        config.get("swing_position_multiplier", 0.85),
                        config.get("swing_orientation_multiplier", 0.85),
                    )

        self.prev_human_foot_pos = {
            side: pos.copy() for side, pos in foot_positions.items()
        }
        self.last_support_state = support_state

    def _apply_stability_weighting(self):
        config = self.retarget_options["stability"]
        metrics = self._compute_com_support_metrics(config)
        self.last_stability_metrics = metrics
        margin = metrics.get("support_margin")
        if margin is None:
            return

        threshold = float(config.get("margin_threshold", 0.02))
        critical = float(config.get("critical_margin", -0.02))
        if margin >= threshold:
            return

        denom = max(threshold - critical, 1e-6)
        severity = float(np.clip((threshold - margin) / denom, 0.0, 1.0))
        torso_multiplier = 1.0 + (
            float(config.get("torso_multiplier", 1.25)) - 1.0
        ) * severity
        waist_multiplier = 1.0 + (
            float(config.get("waist_multiplier", 1.2)) - 1.0
        ) * severity
        arm_multiplier = 1.0 - (
            1.0 - float(config.get("arm_multiplier", 0.85))
        ) * severity

        for info in self.all_task_infos:
            frame_name = info["frame_name"]
            if "torso" in frame_name:
                self._scale_task_cost(info, torso_multiplier, torso_multiplier)
            elif "waist" in frame_name:
                self._scale_task_cost(info, waist_multiplier, waist_multiplier)
            elif any(token in frame_name for token in ("shoulder", "elbow", "wrist")):
                self._scale_task_cost(info, arm_multiplier, arm_multiplier)

    def _compute_com_support_metrics(self, config):
        mj.mj_forward(self.model, self.configuration.data)
        masses = self.model.body_mass
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            return {"support_margin": None, "com_xy": None, "support_feet": []}

        com = np.sum(self.configuration.data.xipos * masses[:, None], axis=0) / total_mass
        support_feet = self._robot_support_feet(config)
        support_points = []
        for foot_body in support_feet:
            support_points.extend(
                self._foot_support_corners(
                    foot_body,
                    float(config.get("foot_length", 0.20)),
                    float(config.get("foot_width", 0.11)),
                )
            )

        if len(support_points) < 3:
            return {
                "support_margin": None,
                "com_xy": com[:2].copy(),
                "support_feet": support_feet,
            }

        hull = self._convex_hull_xy(np.asarray(support_points, dtype=float))
        margin = self._signed_margin_to_polygon(com[:2], hull)
        return {
            "support_margin": float(margin),
            "com_xy": com[:2].copy(),
            "support_feet": support_feet,
            "support_polygon": hull,
        }

    def _robot_support_feet(self, config):
        foot_bodies = [
            config.get("left_foot_body", "l_ankle_x_link"),
            config.get("right_foot_body", "r_ankle_x_link"),
        ]
        support_height = float(config.get("support_height", 0.08))
        heights = {}
        for body_name in foot_bodies:
            body_id = self.robot_body_names.get(body_name)
            if body_id is not None:
                heights[body_name] = float(self.configuration.data.xpos[body_id][2])
        if not heights:
            return []
        min_height = min(heights.values())
        support_feet = [
            body_name
            for body_name, height in heights.items()
            if height <= min_height + support_height
        ]
        return support_feet

    def _foot_support_corners(self, body_name, foot_length, foot_width):
        body_id = self.robot_body_names.get(body_name)
        if body_id is None:
            return []
        center = self.configuration.data.xpos[body_id]
        rot = self.configuration.data.xmat[body_id].reshape(3, 3)
        x_axis = rot[:, 0].copy()
        y_axis = rot[:, 1].copy()
        x_axis[2] = 0.0
        y_axis[2] = 0.0
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(y_axis) < 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        half_length = 0.5 * foot_length
        half_width = 0.5 * foot_width
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                point = center + sx * half_length * x_axis + sy * half_width * y_axis
                corners.append(point[:2].copy())
        return corners

    @staticmethod
    def _convex_hull_xy(points):
        points = sorted({(float(x), float(y)) for x, y in points})
        if len(points) <= 1:
            return np.asarray(points, dtype=float)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return np.asarray(lower[:-1] + upper[:-1], dtype=float)

    @staticmethod
    def _signed_margin_to_polygon(point, polygon):
        if polygon is None or len(polygon) < 3:
            return None
        point = np.asarray(point, dtype=float)
        min_signed_distance = np.inf
        for i in range(len(polygon)):
            a = polygon[i]
            b = polygon[(i + 1) % len(polygon)]
            edge = b - a
            edge_norm = float(np.linalg.norm(edge))
            if edge_norm < 1e-9:
                continue
            signed_distance = (
                edge[0] * (point[1] - a[1]) - edge[1] * (point[0] - a[0])
            ) / edge_norm
            min_signed_distance = min(min_signed_distance, signed_distance)
        if not np.isfinite(min_signed_distance):
            return None
        return float(min_signed_distance)

    def _solve_ik_velocity(self, tasks, dt):
        try:
            vel = mink.solve_ik(
                self.configuration,
                tasks,
                dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
            )
        except Exception as exc:
            collision_config = self.retarget_options["collision_avoidance"]
            if (
                self.collision_avoidance_limit is not None
                and collision_config.get("fail_open", True)
            ):
                self.collision_solve_failures += 1
                if self.verbose and self.collision_solve_failures <= 3:
                    print(
                        "[GMR] Collision-aware IK failed; retry without collision "
                        f"avoidance. Original error: {exc}"
                    )
                fallback_limits = [
                    limit
                    for limit in self.ik_limits
                    if limit is not self.collision_avoidance_limit
                ]
                vel = mink.solve_ik(
                    self.configuration,
                    tasks,
                    dt,
                    self.solver,
                    self.damping,
                    limits=fallback_limits,
                )
            else:
                raise

        vel_array = np.asarray(vel, dtype=float)
        stat_indices = self._joint_velocity_stat_indices()
        if len(stat_indices) > 0:
            vel_array = vel_array[stat_indices]
        abs_vel = np.abs(vel_array)
        self.last_ik_velocity_stats = {
            "max": float(np.max(abs_vel)) if abs_vel.size else 0.0,
            "mean": float(np.mean(abs_vel)) if abs_vel.size else 0.0,
            "p95": float(np.percentile(abs_vel, 95)) if abs_vel.size else 0.0,
        }
        return vel

    def _joint_velocity_stat_indices(self):
        if hasattr(self, "_cached_joint_velocity_stat_indices"):
            return self._cached_joint_velocity_stat_indices

        indices = []
        for joint_name in self.robot_dof_names.keys():
            if not joint_name:
                continue
            joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == mj.mjtJoint.mjJNT_FREE:
                continue
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            if joint_type == mj.mjtJoint.mjJNT_BALL:
                dof_width = 3
            else:
                dof_width = 1
            indices.extend(range(dof_adr, dof_adr + dof_width))

        self._cached_joint_velocity_stat_indices = np.asarray(
            sorted(set(indices)), dtype=np.int32
        )
        return self._cached_joint_velocity_stat_indices

    def _solve_task_stage(self, tasks, error_fn):
        curr_error = error_fn()
        dt = self.configuration.model.opt.timestep
        vel = self._solve_ik_velocity(tasks, dt)
        self.configuration.integrate_inplace(vel, dt)
        next_error = error_fn()
        num_iter = 0
        max_iter = self._max_extra_iterations_for_stage()
        while curr_error - next_error > 0.001 and num_iter < max_iter:
            curr_error = next_error
            dt = self.configuration.model.opt.timestep
            vel = self._solve_ik_velocity(tasks, dt)
            self.configuration.integrate_inplace(vel, dt)
            next_error = error_fn()
            num_iter += 1

    def _max_extra_iterations_for_stage(self):
        velocity_config = self.retarget_options["velocity_limits"]
        configured = velocity_config.get("max_extra_iterations")
        if velocity_config.get("enabled", False) and configured is not None:
            return max(0, int(configured))
        return self.max_iter

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)
        self._apply_dynamic_task_weights()

        if self.use_ik_match_table1:
            self._solve_task_stage(self.tasks1, self.error1)

        if self.use_ik_match_table2:
            self._solve_task_stage(self.tasks2, self.error2)
                
        if self.last_ik_velocity_stats:
            self.ik_velocity_stats_history.append(dict(self.last_ik_velocity_stats))

        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
