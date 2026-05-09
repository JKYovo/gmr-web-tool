# GMR 源码层优化技术报告

## 1. 目标

本轮优化的目标不是单纯让仿真更好看，而是把一部分“机器人可实现性”约束前移到 GMR retarget 的 IK 求解阶段，让 ELF3 输出动作更少自碰撞、更容易经过后处理变成平滑动作。

当前推荐主线：

```text
原始人体动作
  -> GMR 源码层 collision_avoidance
  -> 原版后处理 soft + v2_foot
  -> 仿真检查 / 指标报告
```

后处理里的 `collision` pipeline 保留为备用工具，不作为当前默认主线。

## 2. 源码改动概览

### 2.1 GeneralMotionRetargeting 新增配置化约束

主要文件：

```text
general_motion_retargeting/motion_retarget.py
```

新增 `retarget_options` 参数，用来从脚本显式开启源码层约束：

```text
velocity_limits
collision_avoidance
support_foot
stability
```

默认全部关闭，所以不加新参数时，Web 和普通 GMR 转换保持原行为。

### 2.2 自碰撞避让

实现方式：

- 使用 `mink.CollisionAvoidanceLimit` 接入 IK QP。
- 从 ELF3 IK config 读取碰撞 pair。
- 支持把 body/link 名称展开成实际 MuJoCo collision geom 名。
- 当前主要覆盖：
  - 上肢 vs 躯干
  - 上肢 vs 髋/大腿
  - 左右手臂互撞
  - 左右脚互撞

相关配置：

```text
general_motion_retargeting/ik_configs/bvh_xsens_to_elf3.json
general_motion_retargeting/ik_configs/smplx_to_elf3.json
```

当前配置默认 `enabled: false`，由脚本参数开启。

### 2.3 速度限制

实现方式：

- 使用 `mink.VelocityLimit`。
- 按关节组设置速度上限：
  - 腰
  - 腿
  - 手臂
  - 手腕
  - 单关节覆盖值

同时记录每帧 IK velocity 统计，用于实验脚本汇总。

注意：本轮还修正了一个关键问题。当前 mink 的 `solve_ik` 参数需要用：

```python
mink.solve_ik(..., limits=self.ik_limits)
```

旧写法把 limits 当作位置参数传入，实际没有按预期生效。

### 2.4 支撑脚动态权重

实现方式：

- 根据人体脚部高度和速度判断支撑脚。
- 支撑脚阶段提高 `l_ankle_x_link / r_ankle_x_link` 的 IK 位置和姿态权重。
- 摆动脚阶段降低脚任务权重，避免腿部动作被钉死。

当前结果显示：这个模块还需要继续调参，暂不建议默认启用。

### 2.5 COM / 支撑区稳定指标与轻量权重

实现方式：

- 用 MuJoCo body mass 计算机器人 COM。
- 估计支撑脚范围。
- 计算 COM 投影到支撑多边形的 margin。
- 第一版不做硬动力学约束，只根据 margin 轻量调整：
  - torso/waist 权重提高
  - 上肢权重降低

当前结论：作为评测指标有价值；作为在线权重调节还需要继续实验。

## 3. 脚本入口

### 3.1 BVH 转 ELF3

文件：

```text
scripts/xsens_bvh_to_robot.py
```

新增参数：

```text
--enable_robot_constraints
--enable_velocity_limit
--enable_collision_avoidance
--enable_support_foot
--enable_stability_weighting
```

推荐当前只单独测试：

```bash
--enable_collision_avoidance
```

### 3.2 GVHMR 转 ELF3

文件：

```text
scripts/gvhmr_to_robot.py
```

同样接入上述源码层约束参数。

### 3.3 三阶段实验脚本

新增：

```text
scripts/retarget_bvh_constraints_experiment.py
```

用于无窗口批量生成：

```text
robot_motion_baseline.pkl
robot_motion_collision.pkl
robot_motion_collision_post_soft_v2_foot.pkl
quality_*.json
stability_*.json / stability_*.csv
summary.csv
```

当前脚本会按类型分目录保存：

```text
motions/robot_motion_*.pkl
quality/quality_*.json
stability/stability_*.json / stability_*.csv
summary.csv
experiment_meta.json
```

典型命令：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/retarget_bvh_constraints_experiment.py \
  --robot elf3 \
  --bvh_file "runtime/jobs/水手-001_7540f71f/input.bvh" \
  --reset_to_zero \
  --modes baseline collision \
  --postprocess_modes collision
```

## 4. 可视化增强

文件：

```text
general_motion_retargeting/robot_motion_viewer.py
scripts/vis_robot_motion.py
scripts/vis_robot_motion_dataset.py
```

新增自碰撞显示：

```bash
--show_self_collision
```

默认模式：

- 机器人保持不透明。
- 发生碰撞的 collision geom 变红/橙。
- 碰撞接触点显示紫色小球。
- 碰撞 geom 中心之间显示紫色连线。

可选透明模式：

```bash
--collision_visual_mode transparent --collision_robot_alpha 0.35
```

## 5. 实验结果

测试动作：

```text
runtime/jobs/水手-001_7540f71f/input.bvh
```

实验输出：

```text
runtime/experiments/collision_postprocess/sailor_full/summary.csv
```

三阶段结果：

| 阶段 | 自碰撞帧比例 | 最大穿透 | 脚滑 p95 | 速度 max | 加速度 max | jerk max |
|---|---:|---:|---:|---:|---:|---:|
| 原始 GMR | 36.77% | 6.10 cm | 0.837 m/s | 300.98 | 57147.85 | 10099691.95 |
| GMR + collision | 3.37% | 1.63 cm | 0.844 m/s | 300.98 | 62597.80 | 12723786.89 |
| GMR + collision + 后处理 | 4.36% | 1.86 cm | 0.649 m/s | 7.00 | 210.78 | 16843.59 |

## 6. 结果分析

### 6.1 自碰撞避让有效

源码层 collision 将自碰撞帧比例从 `36.77%` 降到 `3.37%`，最大穿透从 `6.10 cm` 降到 `1.63 cm`。这说明把碰撞避让放进 GMR 的 IK 阶段是有效的。

### 6.2 单独源码避碰会带来高频抖动

GMR + collision 后，自碰撞明显下降，但加速度和 jerk 上升。视觉上会出现“硬拉开手臂”的效果，说明 collision constraint 会在局部对 IK 产生较强推力。

### 6.3 后处理适合做时间序列降抖

经过原版 `soft + v2_foot` 后处理后：

- 速度峰值降到 `7.00`
- 加速度峰值降到 `210.78`
- jerk 峰值降到 `16843.59`
- 脚滑 p95 从 `0.844` 降到 `0.649`

代价是自碰撞略回升到 `4.36%`，但仍远低于原始 GMR 的 `36.77%`。

## 7. 当前推荐方案

当前最推荐的工程路线：

```text
GMR 里开启 collision_avoidance
后处理使用 soft + v2_foot
不默认开启 velocity/support/stability 全约束
```

原因：

- 碰撞是空间几何约束，适合放在 IK 阶段。
- 抖动、速度、加速度、jerk 是时间序列问题，适合后处理。
- 全约束同时开启时，容易牺牲脚滑和 COM/support 指标。

## 8. 暂不推荐默认启用的部分

### 8.1 全约束模式

`--enable_robot_constraints` 会同时打开 velocity、collision、support、stability。当前实验看，全约束能进一步压碰撞，但容易导致脚滑和支撑 margin 变差，不建议作为默认。

### 8.2 后处理 collision pipeline

`tools/motion_postprocess.py --pipeline collision` 保留为备用工具：

- 适合只有 pkl、没有原始 BVH/GVHMR 时补救。
- 适合做对照实验。
- 不建议作为主线，因为它是在结果上硬改，容易引入轻微抖动。

## 9. 下一步计划

1. 继续调 `collision_avoidance` pair 和距离参数，降低“硬拉开”的力度。
2. 将手臂与躯干碰撞 pair 分级：肩/上臂、肘、腕使用不同 detection distance。
3. 给 collision constraint 加阶段性权重或阈值，避免无碰撞时提前过度避让。
4. 单独实验 support foot，不和 velocity/stability 一起开。
5. 将 COM/support 稳定指标用于评估优先，不急着做硬约束。
6. 建立固定动作集批量报告，选出适合真机的默认参数。

## 10. 结论

本轮优化验证了一个明确方向：

```text
自碰撞避免应前置到 GMR 源码层；
平滑、限速、jerk 控制应继续交给后处理。
```

对 ELF3 跳舞动作来说，目前最稳妥的组合是：

```text
GMR + collision_avoidance
  -> soft + v2_foot 后处理
  -> 自碰撞可视化检查
  -> quality/stability 指标检查
```

这条路线已经显著降低自碰撞，同时把源码避碰带来的抖动压到更可接受的范围。

## 11. 常用指令

以下命令都默认在 GMR 仓库目录下运行：

```bash
cd /home/user-kevien/gvhmr_pkg/GMR
```

### 11.1 BVH 使用优化后的 GMR 生成 pkl

推荐当前只开启源码层自碰撞避让：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/xsens_bvh_to_robot.py \
  --robot elf3 \
  --bvh_file "/path/to/motion.bvh" \
  --reset_to_zero \
  --enable_collision_avoidance \
  --save_path "runtime/jobs/xxx/manual_trials/motions/robot_motion_collision.pkl"
```

说明：

- `--enable_collision_avoidance`：只开启 GMR 源码层自碰撞避让。
- `--save_path`：保存优化后的 `robot_motion.pkl`。
- 不加 `--record_video` 就不会生成 mp4。
- 这里省略的常用默认值是：`--bvh_format 3DSM`、`--scale 0.01`、`--ground_clearance 0.03`、`--smoothing_alpha 0.35`。
- `--reset_to_zero` 当前不是默认值，建议日常 BVH 转换保留。

暂不推荐直接使用：

```bash
--enable_robot_constraints
```

因为它会同时打开 velocity、collision、support、stability，当前容易让脚滑和支撑指标变差。

### 11.2 GVHMR 结果使用优化后的 GMR

GVHMR 的 `hmr4d_results.pt` 转 ELF3：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/gvhmr_to_robot.py \
  --robot elf3 \
  --gvhmr_pred_file "/path/to/hmr4d_results.pt" \
  --enable_collision_avoidance \
  --save_path "runtime/jobs/xxx/manual_trials/motions/robot_motion_collision.pkl"
```

### 11.3 对优化后的 GMR 结果做原版后处理

常规舞蹈动作推荐后处理组合：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python tools/motion_postprocess.py optimize \
  --input "runtime/jobs/xxx/manual_trials/motions/robot_motion_collision.pkl" \
  --robot elf3 \
  --profile soft \
  --pipeline v2_foot \
  --output "runtime/jobs/xxx/manual_trials/motions/robot_motion_collision_post_soft_v2_foot.pkl" \
  --quality-json "runtime/jobs/xxx/manual_trials/quality/quality_collision_post_soft_v2_foot.json"
```

说明：

- 不加 `--render`：只生成 pkl 和质量报告，不生成 mp4。
- `soft + v2_foot`：主要负责降速度、降加速度、降 jerk，并轻量处理脚底。
- 如果动作包含明显起跳、落地或单脚支撑，优先改用 `--pipeline arm_only`。它只平滑肩、肘、腕，不改 root 和下肢，避免把脚底接触节奏磨坏。
- 不建议主线使用 `--pipeline collision`，它保留为只有 pkl 时的备用修复工具。

跳跃/落地动作示例：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python tools/motion_postprocess.py optimize \
  --input "runtime/jobs/xxx/manual_trials/motions/robot_motion_collision.pkl" \
  --robot elf3 \
  --profile soft \
  --pipeline arm_only \
  --output "runtime/jobs/xxx/manual_trials/motions/robot_motion_collision_post_arm_only.pkl" \
  --quality-json "runtime/jobs/xxx/manual_trials/quality/quality_collision_post_arm_only.json"
```

### 11.4 只做质量评测

对任意 `robot_motion.pkl` 生成质量报告：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python tools/motion_postprocess.py quality \
  --input "runtime/jobs/xxx/robot_motion.pkl" \
  --robot elf3 \
  --output "runtime/jobs/xxx/motion_quality.json"
```

报告重点看：

```text
self_collision.collision_frame_ratio
self_collision.max_penetration_m
contact.estimated_foot_sliding_speed.p95
dof_velocity.max
dof_acceleration.max
dof_jerk.max
```

### 11.5 稳定性 / COM / 支撑区评测

对已有 pkl 生成 COM 和支撑脚报告：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/diagnose_robot_stability.py \
  --robot elf3 \
  --motion_path "runtime/jobs/xxx/robot_motion.pkl"
```

默认输出：

```text
robot_motion_stability.json
robot_motion_stability.csv
```

重点看：

```text
outside_support_percent
min_support_margin_m
p95_abs_com_forward_from_support_m
p95_abs_torso_forward_lean_deg
p95_abs_waist_forward_lean_deg
```

### 11.6 仿真播放已有 pkl

普通播放：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/vis_robot_motion.py \
  --robot elf3 \
  --robot_motion_path "runtime/jobs/xxx/robot_motion.pkl"
```

默认会显示：

- camera track
- COM 投影
- 支撑脚范围

### 11.7 仿真显示自碰撞区域

不透明模式，推荐默认使用：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/vis_robot_motion.py \
  --robot elf3 \
  --robot_motion_path "runtime/jobs/xxx/robot_motion.pkl" \
  --show_self_collision
```

显示规则：

- 碰撞 geom：红 / 橙色。
- 碰撞接触点：紫色小球。
- 碰撞 geom 中心连线：紫色线。
- 机器人保持不透明，方便判断手臂是在躯干前面还是穿进去了。

透明模式：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/vis_robot_motion.py \
  --robot elf3 \
  --robot_motion_path "runtime/jobs/xxx/robot_motion.pkl" \
  --show_self_collision \
  --collision_visual_mode transparent \
  --collision_robot_alpha 0.35
```

透明模式适合看身体内部，但前后遮挡关系会更难判断。

### 11.8 三阶段完整实验

一次性生成：

```text
原始 GMR
GMR + collision
GMR + collision + soft/v2_foot 后处理
```

命令：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/retarget_bvh_constraints_experiment.py \
  --robot elf3 \
  --bvh_file "/path/to/motion.bvh" \
  --reset_to_zero \
  --modes baseline collision \
  --postprocess_modes collision \
  --output_dir "runtime/experiments/collision_postprocess/xxx"
```

输出：

```text
motions/robot_motion_baseline.pkl
motions/robot_motion_collision.pkl
motions/robot_motion_collision_post_soft_v2_foot.pkl
quality/quality_baseline.json
quality/quality_collision.json
quality/quality_collision_post_soft_v2_foot.json
stability/stability_baseline.json / stability/stability_baseline.csv
stability/stability_collision.json / stability/stability_collision.csv
stability/stability_collision_post_soft_v2_foot.json / stability/stability_collision_post_soft_v2_foot.csv
summary.csv
```

`summary.csv` 是最重要的汇总文件，用来比较三阶段指标。

这个实验脚本默认后处理就是 `soft + v2_foot`，所以常用命令里不用再写
`--postprocess_profile soft` 和 `--postprocess_pipeline v2_foot`。

### 11.9 当前水手实验复现命令

完整水手动作三阶段实验：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/retarget_bvh_constraints_experiment.py \
  --robot elf3 \
  --bvh_file "runtime/jobs/水手-001_7540f71f/input.bvh" \
  --reset_to_zero \
  --modes baseline collision \
  --postprocess_modes collision \
  --output_dir "runtime/experiments/collision_postprocess/sailor_full"
```

播放最终推荐结果并显示碰撞：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n gmr \
python scripts/vis_robot_motion.py \
  --robot elf3 \
  --robot_motion_path "runtime/experiments/collision_postprocess/sailor_full/motions/robot_motion_collision_post_soft_v2_foot.pkl" \
  --show_self_collision
```
