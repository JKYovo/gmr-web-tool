from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import gradio as gr

from gmr_web.common import SOURCE_REGISTRY, SUPPORTED_ROBOTS, TERMINAL_STATUSES, UPLOAD_SOURCE_TYPES


LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def _logs(job):
    return "\n".join(job.get("logs", []))


def _files(job):
    artifacts = job.get("artifacts", {})
    return (
        artifacts.get("motion_path"),
        artifacts.get("video_path"),
        artifacts.get("artifacts_zip_path"),
    )


def _status_text(job):
    if job["status"] == "queued":
        return "任务已入队"
    if job["status"] == "running":
        return "任务处理中"
    if job["status"] == "failed":
        return f"任务失败：{job.get('error_summary') or '未知错误'}"
    if job["status"] == "cancelled":
        return "任务已取消"
    return "任务完成"


def _path_name(value):
    if not value:
        return ""
    return Path(str(value)).name


def _display_file_name(job):
    for key in ("display_name", "source_input_file", "input_file"):
        name = _path_name(job.get(key))
        if name:
            return name
    return "-"


def _display_time(value):
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


def _gvhmr_display_name(job):
    for key in ("display_name", "source_video_path", "source_input_file", "input_video", "video_path"):
        name = _path_name(job.get(key))
        if name:
            return name
    return _path_name(job.get("output_dir")) or job.get("job_id", "GVHMR 结果")


def _format_job_detail(job):
    if not job:
        return ""
    lines = [
        f"job_id: {job['job_id']}",
        f"status: {job['status']}",
        f"source_type: {job['source_type']}",
        f"robot: {job['robot']}",
        f"ik_config: {job.get('ik_config')}",
        f"display_name: {_display_file_name(job)}",
        f"submitted_at: {_display_time(job.get('submitted_at'))}",
        f"source_input_file: {job.get('source_input_file')}",
        f"staged_input_file: {job['input_file']}",
        f"output_dir: {job['output_dir']}",
        f"ground_clearance: {job['ground_clearance']}",
        f"smoothing_alpha: {job['smoothing_alpha']}",
        f"generate_video: {job['generate_video']}",
        f"error_summary: {job.get('error_summary')}",
    ]
    return "\n".join(lines)


def _gvhmr_result_choices(gvhmr_manager):
    if gvhmr_manager is None:
        return []
    choices = []
    for job in gvhmr_manager.list_jobs(limit=50):
        hmr4d_path = job.get("artifacts", {}).get("hmr4d_results_path")
        if job.get("status") != "succeeded" or not hmr4d_path or not Path(hmr4d_path).exists():
            continue
        video_name = _gvhmr_display_name(job)
        label = f"{video_name} | {job['job_id']} | {_display_time(job.get('submitted_at'))}"
        choices.append((label, job["job_id"]))
    return choices


def _format_gvhmr_result(job):
    if not job:
        return ""
    hmr4d_path = job.get("artifacts", {}).get("hmr4d_results_path")
    lines = [
        f"GVHMR job_id: {job['job_id']}",
        f"status: {job['status']}",
        f"video: {_gvhmr_display_name(job)}",
        f"submitted_at: {_display_time(job.get('submitted_at'))}",
        f"hmr4d_results.pt: {hmr4d_path or '缺失'}",
        f"output_dir: {job['output_dir']}",
    ]
    return "\n".join(lines)


def build_ui(manager, gvhmr_manager=None):
    source_labels = {
        "auto": "auto 自动识别",
        **{
            name: f"{name} - {spec['label']}"
            for name, spec in SOURCE_REGISTRY.items()
            if spec["upload_enabled"]
        },
    }
    source_choices = [(source_labels[name], name) for name in UPLOAD_SOURCE_TYPES]
    config_rows = [
        [
            name,
            spec["label"],
            ", ".join(spec["extensions"]) or "专用采集/实时输入",
            "可上传" if spec["upload_enabled"] else "已配置，暂不开放上传",
            spec["ik_config"],
        ]
        for name, spec in SOURCE_REGISTRY.items()
    ]
    gvhmr_choices = _gvhmr_result_choices(gvhmr_manager)
    gvhmr_initial = gvhmr_choices[0][1] if gvhmr_choices else None

    with gr.Blocks(title="GMR Web") as app:
        gr.Markdown("# GMR Web")
        gr.Markdown("内部 ELF3 机器人动作转换工具：上传人体动作文件，后台生成机器人 `pkl` 和可选预览视频。")

        if gvhmr_manager is not None:
            with gr.Tab("从 GVHMR 结果转换"):
                gr.Markdown("选择刚刚由 GVHMR 生成的结果，直接提交 ELF3 转换，不需要手动查找 `hmr4d_results.pt`。")
                with gr.Row():
                    refresh_gvhmr = gr.Button("刷新 GVHMR 结果")
                    gvhmr_selector = gr.Dropdown(
                        label="选择 GVHMR 成功任务",
                        choices=gvhmr_choices,
                        value=gvhmr_initial,
                    )
                    submit_gvhmr = gr.Button("使用该结果转 ELF3", variant="primary")
                gvhmr_detail = gr.Textbox(label="GVHMR 结果详情", lines=5, interactive=False)
                gvhmr_status = gr.Textbox(label="任务状态", interactive=False)
                gvhmr_gmr_job_id = gr.Textbox(label="GMR 任务 ID", interactive=False)
                gvhmr_output_dir = gr.Textbox(label="GMR 输出目录", interactive=False)
                gvhmr_logs = gr.Textbox(label="GMR 任务日志", lines=10, interactive=False)
                with gr.Row():
                    gvhmr_motion_file = gr.File(label="robot_motion.pkl")
                    gvhmr_video_file = gr.Video(label="robot_preview.mp4")
                    gvhmr_artifact_file = gr.File(label="artifacts.zip")

        with gr.Tab("单文件转换"):
            with gr.Row():
                input_file = gr.File(label="上传 hmr4d_results.pt / SMPL-X npz / BVH / fbx_offline pkl", file_count="single", type="filepath")
                with gr.Column():
                    source_type = gr.Dropdown(source_choices, value="auto", label="输入类型")
                    robot = gr.Dropdown(list(SUPPORTED_ROBOTS), value="elf3", label="机器人")
                    ground_clearance = gr.Number(value=0.03, label="离地安全量 (m)")
                    smoothing_alpha = gr.Slider(0.05, 1.0, value=0.35, step=0.05, label="BVH 下肢平滑系数")
                    generate_video = gr.Checkbox(value=True, label="生成预览视频")
                    submit = gr.Button("提交转换", variant="primary")

            status = gr.Textbox(label="任务状态", interactive=False)
            job_id = gr.Textbox(label="任务 ID", interactive=False)
            output_dir = gr.Textbox(label="输出目录", interactive=False)
            logs = gr.Textbox(label="任务日志", lines=10, interactive=False)
            with gr.Row():
                motion_file = gr.File(label="robot_motion.pkl")
                video_file = gr.Video(label="robot_preview.mp4")
                artifact_file = gr.File(label="artifacts.zip")

        with gr.Tab("任务历史"):
            refresh = gr.Button("刷新历史")
            with gr.Row():
                status_filter = gr.Dropdown(
                    ["全部", "queued", "running", "succeeded", "failed", "cancelled"],
                    value="全部",
                    label="状态筛选",
                )
                source_filter = gr.Dropdown(
                    ["全部"] + list(SOURCE_REGISTRY.keys()),
                    value="全部",
                    label="输入类型筛选",
                )
            jobs_table = gr.Dataframe(headers=["job_id", "file", "source", "status", "submitted_at"], interactive=False)
            selected_job = gr.Textbox(label="输入任务 ID 查看详情")
            load_job = gr.Button("查看任务")
            retry_job = gr.Button("重试任务")
            cancel_job = gr.Button("取消任务")
            history_status = gr.Textbox(label="历史状态", interactive=False)
            job_detail = gr.Textbox(label="任务详情", lines=10, interactive=False)
            history_logs = gr.Textbox(label="任务日志", lines=10, interactive=False)
            with gr.Row():
                history_motion = gr.File(label="robot_motion.pkl")
                history_video = gr.Video(label="robot_preview.mp4")
                history_zip = gr.File(label="artifacts.zip")

        with gr.Tab("配置说明"):
            gr.Markdown("## ELF3 输入配置")
            gr.Dataframe(
                value=config_rows,
                headers=["source_type", "说明", "文件类型", "状态", "IK 配置"],
                interactive=False,
            )
            gr.Markdown("默认参数：`ground_clearance=0.03`，`smoothing_alpha=0.35`，生成 H.264 `robot_preview.mp4`。")

        def submit_job(file_path, src_type, robot_name, clearance, smooth_alpha, make_video):
            if not file_path:
                yield ("请先上传一个 pt 或 bvh 文件。", "", "", "", None, None, None)
                return
            try:
                job = manager.submit_job(
                    input_file=file_path,
                    source_type=src_type,
                    robot=robot_name,
                    ground_clearance=clearance,
                    smoothing_alpha=smooth_alpha,
                    generate_video=make_video,
                )
            except Exception as exc:
                yield (f"提交失败：{exc}", "", "", "", None, None, None)
                return

            while True:
                current = manager.get_job(job["job_id"])
                motion, video, zip_file = _files(current)
                yield (
                    _status_text(current),
                    current["job_id"],
                    current["output_dir"],
                    _logs(current),
                    motion,
                    video,
                    zip_file,
                )
                if current["status"] in TERMINAL_STATUSES:
                    break
                import time

                time.sleep(1)

        def refresh_gvhmr_results():
            choices = _gvhmr_result_choices(gvhmr_manager)
            value = choices[0][1] if choices else None
            detail = "暂无可转换的 GVHMR 成功任务。"
            if value:
                detail = _format_gvhmr_result(gvhmr_manager.get_job(value))
            return gr.update(choices=choices, value=value), detail

        def describe_gvhmr_result(gvhmr_job_id):
            if not gvhmr_job_id:
                return "请选择一个 GVHMR 成功任务。"
            job = gvhmr_manager.get_job(gvhmr_job_id)
            if job is None:
                return f"GVHMR 任务不存在：{gvhmr_job_id}"
            return _format_gvhmr_result(job)

        def submit_gvhmr_result(gvhmr_job_id):
            if not gvhmr_job_id:
                yield ("请先选择一个 GVHMR 成功任务。", "", "", "", None, None, None)
                return
            gvhmr_job = gvhmr_manager.get_job(gvhmr_job_id)
            if gvhmr_job is None:
                yield (f"GVHMR 任务不存在：{gvhmr_job_id}", "", "", "", None, None, None)
                return
            if gvhmr_job.get("status") != "succeeded":
                yield ("GVHMR 任务还没有成功完成，不能转 ELF3。", "", "", "", None, None, None)
                return
            hmr4d_path = gvhmr_job.get("artifacts", {}).get("hmr4d_results_path")
            if not hmr4d_path or not Path(hmr4d_path).exists():
                yield ("该 GVHMR 任务缺少 hmr4d_results.pt，不能转 ELF3。", "", "", "", None, None, None)
                return

            try:
                job = manager.submit_job(
                    input_file=hmr4d_path,
                    source_type="gvhmr_smplx",
                    robot="elf3",
                    ground_clearance=0.03,
                    smoothing_alpha=0.35,
                    generate_video=True,
                    display_name=_gvhmr_display_name(gvhmr_job),
                )
            except Exception as exc:
                yield (f"提交失败：{exc}", "", "", "", None, None, None)
                return

            while True:
                current = manager.get_job(job["job_id"])
                motion, video, zip_file = _files(current)
                yield (
                    _status_text(current),
                    current["job_id"],
                    current["output_dir"],
                    _logs(current),
                    motion,
                    video,
                    zip_file,
                )
                if current["status"] in TERMINAL_STATUSES:
                    break
                import time

                time.sleep(1)

        def refresh_history(status_value, source_value):
            rows = []
            for job in manager.list_jobs(limit=50):
                if status_value != "全部" and job["status"] != status_value:
                    continue
                if source_value != "全部" and job["source_type"] != source_value:
                    continue
                rows.append(
                    [
                        job["job_id"],
                        _display_file_name(job),
                        job["source_type"],
                        job["status"],
                        _display_time(job.get("submitted_at")),
                    ]
                )
            return rows

        def inspect_job(job_id):
            job_id = (job_id or "").strip()
            if not job_id:
                return ("请输入任务 ID。", "", "", None, None, None)
            job = manager.get_job(job_id)
            if job is None:
                return (f"任务不存在：{job_id}", "", "", None, None, None)
            motion, video, zip_file = _files(job)
            return (_status_text(job), _format_job_detail(job), _logs(job), motion, video, zip_file)

        def retry_selected(job_id):
            job_id = (job_id or "").strip()
            if not job_id:
                return ("请输入任务 ID。", "", "", None, None, None)
            try:
                job = manager.retry_job(job_id)
            except Exception as exc:
                return (f"重试失败：{exc}", "", "", None, None, None)
            if job is None:
                return (f"任务不存在：{job_id}", "", "", None, None, None)
            return inspect_job(job_id)

        def cancel_selected(job_id):
            job_id = (job_id or "").strip()
            if not job_id:
                return ("请输入任务 ID。", "", "", None, None, None)
            job = manager.cancel_job(job_id)
            if job is None:
                return (f"任务不存在：{job_id}", "", "", None, None, None)
            return inspect_job(job_id)

        submit.click(
            submit_job,
            inputs=[input_file, source_type, robot, ground_clearance, smoothing_alpha, generate_video],
            outputs=[status, job_id, output_dir, logs, motion_file, video_file, artifact_file],
        )
        if gvhmr_manager is not None:
            refresh_gvhmr.click(refresh_gvhmr_results, outputs=[gvhmr_selector, gvhmr_detail])
            gvhmr_selector.change(describe_gvhmr_result, inputs=[gvhmr_selector], outputs=[gvhmr_detail])
            submit_gvhmr.click(
                submit_gvhmr_result,
                inputs=[gvhmr_selector],
                outputs=[
                    gvhmr_status,
                    gvhmr_gmr_job_id,
                    gvhmr_output_dir,
                    gvhmr_logs,
                    gvhmr_motion_file,
                    gvhmr_video_file,
                    gvhmr_artifact_file,
                ],
            )
        refresh.click(refresh_history, inputs=[status_filter, source_filter], outputs=[jobs_table])
        load_job.click(
            inspect_job,
            inputs=[selected_job],
            outputs=[history_status, job_detail, history_logs, history_motion, history_video, history_zip],
        )
        retry_job.click(
            retry_selected,
            inputs=[selected_job],
            outputs=[history_status, job_detail, history_logs, history_motion, history_video, history_zip],
        )
        cancel_job.click(
            cancel_selected,
            inputs=[selected_job],
            outputs=[history_status, job_detail, history_logs, history_motion, history_video, history_zip],
        )

    return app.queue()
