import html
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import gradio as gr

from gmr_web.common import SOURCE_REGISTRY, SUPPORTED_ROBOTS, TERMINAL_STATUSES, UPLOAD_SOURCE_TYPES


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
RENDER_LOG_RE = re.compile(r"^\[Render\]\s+(\d+)/(\d+)")
RETARGET_LOG_RE = re.compile(r"^\[GMR\]\s+Retarget\s+(\d+)/(\d+)")
MAX_PROGRESS_LOGS = 10
SELECTABLE_OUTPUT_CSS = """
.gradio-container, .gradio-container * {
  user-select: text !important;
  -webkit-user-select: text !important;
}
textarea, input, pre, code, .cm-content, .cm-line {
  user-select: text !important;
  -webkit-user-select: text !important;
}
.gmr-history-card {
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 14px 16px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
}
.gmr-history-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
}
.gmr-history-item {
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.78);
  padding: 10px 12px;
  border: 1px solid rgba(148, 163, 184, 0.22);
}
.gmr-history-label {
  color: #64748b;
  font-size: 12px;
  margin-bottom: 4px;
}
.gmr-history-value {
  color: #0f172a;
  font-size: 14px;
  font-weight: 650;
  word-break: break-word;
}
.gmr-quality-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px;
}
.gmr-empty-note {
  color: #64748b;
  border: 1px dashed #cbd5e1;
  border-radius: 12px;
  padding: 12px 14px;
  background: #f8fafc;
}
.gmr-hero {
  border: 1px solid #dbeafe;
  border-radius: 20px;
  padding: 20px 22px;
  background:
    radial-gradient(circle at 10% 10%, rgba(14, 165, 233, 0.16), transparent 30%),
    linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
  margin-bottom: 14px;
}
.gmr-hero h1 {
  margin: 0 0 6px;
  font-size: 30px;
  letter-spacing: -0.03em;
}
.gmr-section-title {
  margin: 18px 0 8px;
  color: #0f172a;
  font-size: 18px;
  font-weight: 750;
}
.gmr-download-row {
  align-items: center;
}
.gmr-video-compare video,
.gmr-preview-video video {
  border-radius: 14px;
  background: #0f172a;
}
.gmr-sync-controls {
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 10px;
  background: #f8fafc;
  margin-bottom: 8px;
}
"""

SYNC_PLAY_JS = """
() => {
  const root = document.querySelector('.gmr-video-compare');
  const videos = root ? Array.from(root.querySelectorAll('video')) : [];
  if (videos.length < 2) return [];
  const playable = videos.filter((video) => Number.isFinite(video.duration) && video.duration > 0);
  if (!playable.length) return [];
  const baseTime = playable[0].currentTime || 0;
  const shouldPlay = playable.some((video) => video.paused);
  playable.forEach((video) => {
    if (Math.abs(video.currentTime - baseTime) > 0.20) video.currentTime = baseTime;
    if (shouldPlay) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  });
  return [];
}
"""

SYNC_SEEK_JS = """
() => {
  const root = document.querySelector('.gmr-video-compare');
  const videos = root ? Array.from(root.querySelectorAll('video')) : [];
  if (videos.length < 2) return [];
  const playable = videos.filter((video) => Number.isFinite(video.duration) && video.duration > 0);
  if (!playable.length) return [];
  const baseTime = playable[0].currentTime || 0;
  playable.slice(1).forEach((video) => {
    video.currentTime = Math.min(baseTime, video.duration || baseTime);
  });
  return [];
}
"""

RESET_VIDEOS_JS = """
() => {
  const root = document.querySelector('.gmr-video-compare');
  const videos = root ? Array.from(root.querySelectorAll('video')) : [];
  videos.forEach((video) => {
    video.pause();
    video.currentTime = 0;
  });
  return [];
}
"""

SET_SPEED_JS = """
(rate) => {
  const root = document.querySelector('.gmr-video-compare');
  const videos = root ? Array.from(root.querySelectorAll('video')) : [];
  const speed = Number.parseFloat(rate || 1) || 1;
  videos.forEach((video) => {
    video.playbackRate = speed;
  });
  return [];
}
"""


def _progress_log_key(line):
    line_text = str(line)
    if match := RENDER_LOG_RE.match(line_text):
        return ("render", int(match.group(2)))
    if match := RETARGET_LOG_RE.match(line_text):
        return ("retarget", int(match.group(2)))
    return None


def _sample_progress_lines(lines):
    if len(lines) <= MAX_PROGRESS_LOGS:
        return list(lines), 0
    span = len(lines) - 1
    indices = sorted({round(idx * span / (MAX_PROGRESS_LOGS - 1)) for idx in range(MAX_PROGRESS_LOGS)})
    return [lines[index] for index in indices], len(lines) - len(indices)


def _logs(job):
    logs = job.get("logs", [])
    compacted = []
    progress_buffer = []
    progress_key = None
    skipped_progress_logs = 0

    def flush_progress_buffer():
        nonlocal progress_buffer, progress_key, skipped_progress_logs
        if not progress_buffer:
            return
        sampled, skipped = _sample_progress_lines(progress_buffer)
        compacted.extend(sampled)
        skipped_progress_logs += skipped
        progress_buffer = []
        progress_key = None

    for line in logs:
        key = _progress_log_key(line)
        if key is None:
            flush_progress_buffer()
            compacted.append(line)
            continue
        if progress_buffer and key != progress_key:
            flush_progress_buffer()
        progress_buffer.append(line)
        progress_key = key

    flush_progress_buffer()
    if skipped_progress_logs:
        compacted.append(f"[Progress] 已隐藏 {skipped_progress_logs} 条中间进度日志。")
    return "\n".join(str(line) for line in compacted)


def _files(job):
    artifacts = job.get("artifacts", {})
    return (
        artifacts.get("motion_path"),
        artifacts.get("video_path"),
        artifacts.get("artifacts_zip_path"),
    )


def _postprocess_files(job):
    artifacts = job.get("artifacts", {})
    postprocess_artifacts = job.get("postprocess_artifacts") or {}
    return (
        postprocess_artifacts.get("optimized_motion_path") or artifacts.get("optimized_motion_path"),
        postprocess_artifacts.get("optimized_video_path") or artifacts.get("optimized_video_path"),
        postprocess_artifacts.get("quality_report_path") or artifacts.get("quality_report_path"),
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


def _postprocess_status_text(job):
    status = job.get("postprocess_status")
    if not status:
        return "尚未生成优化版"
    if status == "queued":
        return "后处理已入队"
    if status == "running":
        return "后处理中"
    if status == "failed":
        return f"后处理失败：{job.get('postprocess_error') or '未知错误'}"
    if status == "succeeded":
        return "优化版已生成"
    return str(status)


def _path_name(value):
    if not value:
        return ""
    return Path(str(value)).name


def _safe_html(value):
    return html.escape(str(value if value is not None else "-"))


def _existing_path(value):
    if not value:
        return None
    path = Path(str(value))
    return str(path) if path.exists() else None


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
        f"postprocess_status: {job.get('postprocess_status')}",
        f"postprocess_profile: {job.get('postprocess_profile')}",
        f"postprocess_pipeline: {job.get('postprocess_pipeline')}",
        f"postprocess_render: {job.get('postprocess_render')}",
        f"postprocess_error: {job.get('postprocess_error')}",
    ]
    return "\n".join(lines)


def _format_history_summary(job):
    if not job:
        return "<div class='gmr-empty-note'>请选择一个任务查看详情。</div>"
    source_name = _display_file_name(job)
    output_dir = job.get("output_dir") or "-"
    fields = [
        ("文件名", source_name),
        ("Source", job.get("source_type", "-")),
        ("任务状态", job.get("status", "-")),
        ("后处理", job.get("postprocess_status") or "尚未生成"),
        ("提交时间", _display_time(job.get("submitted_at"))),
        ("输出目录", output_dir),
    ]
    items = "\n".join(
        "<div class='gmr-history-item'>"
        f"<div class='gmr-history-label'>{_safe_html(label)}</div>"
        f"<div class='gmr-history-value'>{_safe_html(value)}</div>"
        "</div>"
        for label, value in fields
    )
    return f"<div class='gmr-history-card'><div class='gmr-history-grid'>{items}</div></div>"


def _format_result_summary(job):
    if not job:
        return "<div class='gmr-empty-note'>提交任务后这里会显示结果摘要。</div>"
    motion, video, zip_file = _files(job)
    fields = [
        ("任务 ID", job.get("job_id", "-")),
        ("文件名", _display_file_name(job)),
        ("状态", job.get("status", "-")),
        ("Source", job.get("source_type", "-")),
        ("输出目录", job.get("output_dir", "-")),
        ("结果", " / ".join(name for name in [_path_name(motion), _path_name(video), _path_name(zip_file)] if name) or "-"),
    ]
    items = "\n".join(
        "<div class='gmr-history-item'>"
        f"<div class='gmr-history-label'>{_safe_html(label)}</div>"
        f"<div class='gmr-history-value'>{_safe_html(value)}</div>"
        "</div>"
        for label, value in fields
    )
    return f"<div class='gmr-history-card'><div class='gmr-history-grid'>{items}</div></div>"


def _run_output_tuple(job):
    motion, video, zip_file = _files(job)
    return (
        _status_text(job),
        job["job_id"],
        job["output_dir"],
        _format_result_summary(job),
        _logs(job),
        _existing_path(video),
        _existing_path(motion),
        _existing_path(video),
        _existing_path(zip_file),
    )


def _empty_run_tuple(message):
    return (
        message,
        "",
        "",
        "<div class='gmr-empty-note'>暂无任务结果。</div>",
        "",
        None,
        None,
        None,
        None,
    )


def _nested_value(data, *keys):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _fmt_metric(value, suffix=""):
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        text = f"{number:.1f}"
    elif abs(number) >= 10:
        text = f"{number:.2f}"
    else:
        text = f"{number:.4f}"
    return f"{text}{suffix}"


def _format_quality_summary(quality_path):
    path = _existing_path(quality_path)
    if not path:
        return "<div class='gmr-empty-note'>暂无优化质量报告。生成优化版后这里会显示 spike、加速度、jerk 和脚滑摘要。</div>"
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return f"<div class='gmr-empty-note'>质量报告读取失败：{_safe_html(exc)}</div>"

    after = data.get("after") if isinstance(data.get("after"), dict) else data
    contact = after.get("contact", {}) if isinstance(after, dict) else {}
    foot_lock = data.get("foot_lock", {}) if isinstance(data.get("foot_lock"), dict) else {}
    repair = data.get("repair", {}) if isinstance(data.get("repair"), dict) else {}
    fields = [
        ("优化器", data.get("optimizer_version", "-")),
        ("Pipeline", data.get("pipeline", "-")),
        ("Spike 数", after.get("spike_count_total", "-") if isinstance(after, dict) else "-"),
        ("加速度 max", _fmt_metric(_nested_value(after, "dof_acceleration", "max"), " rad/s^2")),
        ("Jerk max", _fmt_metric(_nested_value(after, "dof_jerk", "max"), " rad/s^3")),
        ("脚滑 p95", _fmt_metric(_nested_value(contact, "estimated_foot_sliding_speed", "p95"), " m/s")),
        ("脚穿地估计", _fmt_metric(contact.get("estimated_ground_penetration_depth"), " m")),
        ("Foot-lock 帧数", foot_lock.get("corrected_frames", "-")),
        ("修复异常帧", repair.get("spike_frame_count", "-")),
    ]
    items = "\n".join(
        "<div class='gmr-history-item'>"
        f"<div class='gmr-history-label'>{_safe_html(label)}</div>"
        f"<div class='gmr-history-value'>{_safe_html(value)}</div>"
        "</div>"
        for label, value in fields
    )
    return f"<div class='gmr-history-card'><div class='gmr-quality-list'>{items}</div></div>"


def _job_output_tuple(job, status_message=None, postprocess_message=None):
    motion, video, zip_file = _files(job)
    optimized_motion, optimized_video, quality_report = _postprocess_files(job)
    return (
        _format_history_summary(job),
        status_message or _status_text(job),
        postprocess_message or _postprocess_status_text(job),
        _format_quality_summary(quality_report),
        _format_job_detail(job),
        _logs(job),
        _existing_path(video),
        _existing_path(optimized_video),
        _existing_path(motion),
        _existing_path(video),
        _existing_path(optimized_motion),
        _existing_path(optimized_video),
        _existing_path(quality_report),
        _existing_path(zip_file),
    )


def _empty_history_tuple(message):
    return (
        "<div class='gmr-empty-note'>请选择一个任务查看详情。</div>",
        message,
        "",
        "<div class='gmr-empty-note'>暂无优化质量报告。</div>",
        "",
        "",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


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

    with gr.Blocks(title="GMR Web", css=SELECTABLE_OUTPUT_CSS) as app:
        gr.HTML(
            """
            <section class="gmr-hero">
              <h1>GMR Web</h1>
              <p>内部 ELF3 机器人动作转换工具：上传人体动作文件，后台生成机器人 pkl、预览视频和可选优化版。</p>
            </section>
            """
        )

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
                gvhmr_detail = gr.Code(label="GVHMR 结果详情", language="shell", lines=5)
                gvhmr_status = gr.Textbox(label="任务状态", interactive=True, show_copy_button=True)
                gvhmr_gmr_job_id = gr.Textbox(label="GMR 任务 ID", interactive=True, show_copy_button=True)
                gvhmr_output_dir = gr.Textbox(label="GMR 输出目录", interactive=True, show_copy_button=True)
                gvhmr_result_summary = gr.HTML("<div class='gmr-empty-note'>提交任务后这里会显示结果摘要。</div>")
                gvhmr_video_file = gr.Video(
                    label="robot_preview.mp4",
                    height=360,
                    show_download_button=True,
                    elem_classes=["gmr-preview-video"],
                )
                with gr.Row(elem_classes=["gmr-download-row"]):
                    gvhmr_motion_download = gr.DownloadButton("下载 robot_motion.pkl")
                    gvhmr_video_download = gr.DownloadButton("下载 robot_preview.mp4")
                    gvhmr_artifact_download = gr.DownloadButton("下载 artifacts.zip", variant="primary")
                with gr.Accordion("任务日志", open=False):
                    gvhmr_logs = gr.Code(label="GMR 任务日志", language="shell", lines=10)

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

            status = gr.Textbox(label="任务状态", interactive=True, show_copy_button=True)
            job_id = gr.Textbox(label="任务 ID", interactive=True, show_copy_button=True)
            output_dir = gr.Textbox(label="输出目录", interactive=True, show_copy_button=True)
            result_summary = gr.HTML("<div class='gmr-empty-note'>提交任务后这里会显示结果摘要。</div>")
            video_file = gr.Video(
                label="robot_preview.mp4",
                height=360,
                show_download_button=True,
                elem_classes=["gmr-preview-video"],
            )
            with gr.Row(elem_classes=["gmr-download-row"]):
                motion_download = gr.DownloadButton("下载 robot_motion.pkl")
                video_download = gr.DownloadButton("下载 robot_preview.mp4")
                artifact_download = gr.DownloadButton("下载 artifacts.zip", variant="primary")
            with gr.Accordion("任务日志", open=False):
                logs = gr.Code(label="任务日志", language="shell", lines=10)

        with gr.Tab("任务历史"):
            gr.Markdown("## 任务历史")
            with gr.Row():
                refresh = gr.Button("刷新历史", variant="primary")
                job_selector = gr.Dropdown(label="选择任务（不用从表格里复制）", choices=[])
                selected_job = gr.Textbox(
                    label="任务 ID（可复制 / 可手动输入）",
                    interactive=True,
                    show_copy_button=True,
                )
            with gr.Row():
                load_job = gr.Button("查看任务", variant="primary")
                retry_job = gr.Button("重试任务")
                cancel_job = gr.Button("取消任务")
                postprocess_job = gr.Button("生成优化版", variant="primary")

            with gr.Accordion("高级后处理参数", open=False):
                postprocess_profile = gr.Dropdown(
                    ["soft", "preview", "strict"],
                    value="soft",
                    label="平滑强度 profile",
                )
                postprocess_pipeline = gr.Dropdown(
                    [("综合优化 v2_foot（推荐）", "v2_foot"), ("只做关节平滑 v2", "v2")],
                    value="v2_foot",
                    label="优化流程 pipeline",
                )
                postprocess_render = gr.Checkbox(value=True, label="生成优化版预览视频")

            with gr.Accordion("任务列表与筛选", open=False):
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
                jobs_table = gr.Dataframe(
                    headers=["job_id", "file", "source", "status", "postprocess", "submitted_at"],
                    interactive=False,
                )

            gr.Markdown("### 任务摘要")
            history_summary = gr.HTML("<div class='gmr-empty-note'>请选择一个任务查看详情。</div>")
            with gr.Row():
                history_status = gr.Textbox(label="历史状态", interactive=True, show_copy_button=True)
                history_postprocess_status = gr.Textbox(label="后处理状态", interactive=True, show_copy_button=True)

            gr.HTML("<div class='gmr-section-title'>视频对比</div>")
            with gr.Row(elem_classes=["gmr-sync-controls"]):
                sync_play = gr.Button("同步播放 / 暂停", size="sm")
                sync_seek = gr.Button("同步进度", size="sm")
                reset_videos = gr.Button("重置到开头", size="sm")
                playback_speed = gr.Dropdown(
                    [("0.25x", "0.25"), ("0.5x", "0.5"), ("1x", "1"), ("1.5x", "1.5"), ("2x", "2")],
                    value="1",
                    label="播放速度",
                    min_width=120,
                )
            with gr.Row(elem_classes=["gmr-video-compare"]):
                history_video = gr.Video(label="原始预览 robot_preview.mp4", height=360, show_download_button=True)
                history_optimized_video = gr.Video(label="优化预览 preview_foot.mp4", height=360, show_download_button=True)

            gr.HTML("<div class='gmr-section-title'>下载与质量摘要</div>")
            with gr.Row():
                history_motion_download = gr.DownloadButton("下载 robot_motion.pkl")
                history_video_download = gr.DownloadButton("下载 robot_preview.mp4")
                history_optimized_motion_download = gr.DownloadButton("下载 motion_foot.pkl")
            with gr.Row():
                history_optimized_video_download = gr.DownloadButton("下载 preview_foot.mp4")
                history_quality_download = gr.DownloadButton("下载 quality_foot.json")
                history_zip_download = gr.DownloadButton("下载 artifacts.zip", variant="primary")

            with gr.Accordion("质量摘要", open=False):
                history_quality_summary = gr.HTML("<div class='gmr-empty-note'>暂无优化质量报告。</div>")

            with gr.Accordion("高级信息：日志与任务详情", open=False):
                history_logs = gr.Code(label="任务日志", language="shell", lines=10)
                job_detail = gr.Code(label="任务详情", language="shell", lines=10)

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
                yield _empty_run_tuple("请先上传一个 pt 或 bvh 文件。")
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
                yield _empty_run_tuple(f"提交失败：{exc}")
                return

            while True:
                current = manager.get_job(job["job_id"])
                yield _run_output_tuple(current)
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
                yield _empty_run_tuple("请先选择一个 GVHMR 成功任务。")
                return
            gvhmr_job = gvhmr_manager.get_job(gvhmr_job_id)
            if gvhmr_job is None:
                yield _empty_run_tuple(f"GVHMR 任务不存在：{gvhmr_job_id}")
                return
            if gvhmr_job.get("status") != "succeeded":
                yield _empty_run_tuple("GVHMR 任务还没有成功完成，不能转 ELF3。")
                return
            hmr4d_path = gvhmr_job.get("artifacts", {}).get("hmr4d_results_path")
            if not hmr4d_path or not Path(hmr4d_path).exists():
                yield _empty_run_tuple("该 GVHMR 任务缺少 hmr4d_results.pt，不能转 ELF3。")
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
                yield _empty_run_tuple(f"提交失败：{exc}")
                return

            while True:
                current = manager.get_job(job["job_id"])
                yield _run_output_tuple(current)
                if current["status"] in TERMINAL_STATUSES:
                    break
                import time

                time.sleep(1)

        def refresh_history(status_value, source_value):
            rows = []
            choices = []
            for job in manager.list_jobs(limit=50):
                if status_value != "全部" and job["status"] != status_value:
                    continue
                if source_value != "全部" and job["source_type"] != source_value:
                    continue
                label = (
                    f"{_display_file_name(job)} | {job['job_id']} | "
                    f"{job['status']} | {_display_time(job.get('submitted_at'))}"
                )
                choices.append((label, job["job_id"]))
                rows.append(
                    [
                        job["job_id"],
                        _display_file_name(job),
                        job["source_type"],
                        job["status"],
                        job.get("postprocess_status") or "-",
                        _display_time(job.get("submitted_at")),
                    ]
                )
            first_job_id = choices[0][1] if choices else ""
            return rows, gr.update(choices=choices, value=first_job_id or None), first_job_id

        def fill_selected_job(job_id):
            return job_id or ""

        def inspect_job(job_id):
            job_id = (job_id or "").strip()
            if not job_id:
                return _empty_history_tuple("请输入任务 ID。")
            job = manager.get_job(job_id)
            if job is None:
                return _empty_history_tuple(f"任务不存在：{job_id}")
            return _job_output_tuple(job)

        def request_postprocess_selected(job_id, profile, pipeline, render):
            job_id = (job_id or "").strip()
            if not job_id:
                yield _empty_history_tuple("请输入任务 ID。")
                return
            try:
                job = manager.request_postprocess(
                    job_id,
                    profile=profile,
                    pipeline=pipeline,
                    render=render,
                )
            except Exception as exc:
                existing = manager.get_job(job_id)
                if existing is None:
                    yield _empty_history_tuple(f"后处理提交失败：{exc}")
                else:
                    yield _job_output_tuple(existing, postprocess_message=f"后处理提交失败：{exc}")
                return
            if job is None:
                yield _empty_history_tuple(f"任务不存在：{job_id}")
                return

            while True:
                current = manager.get_job(job["job_id"])
                yield _job_output_tuple(current)
                if current.get("postprocess_status") in {"succeeded", "failed"}:
                    break
                import time

                time.sleep(1)

        def retry_selected(job_id):
            job_id = (job_id or "").strip()
            if not job_id:
                return _empty_history_tuple("请输入任务 ID。")
            try:
                job = manager.retry_job(job_id)
            except Exception as exc:
                return _empty_history_tuple(f"重试失败：{exc}")
            if job is None:
                return _empty_history_tuple(f"任务不存在：{job_id}")
            return inspect_job(job_id)

        def cancel_selected(job_id):
            job_id = (job_id or "").strip()
            if not job_id:
                return _empty_history_tuple("请输入任务 ID。")
            job = manager.cancel_job(job_id)
            if job is None:
                return _empty_history_tuple(f"任务不存在：{job_id}")
            return inspect_job(job_id)

        submit.click(
            submit_job,
            inputs=[input_file, source_type, robot, ground_clearance, smoothing_alpha, generate_video],
            outputs=[
                status,
                job_id,
                output_dir,
                result_summary,
                logs,
                video_file,
                motion_download,
                video_download,
                artifact_download,
            ],
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
                    gvhmr_result_summary,
                    gvhmr_logs,
                    gvhmr_video_file,
                    gvhmr_motion_download,
                    gvhmr_video_download,
                    gvhmr_artifact_download,
                ],
            )
        sync_play.click(fn=None, inputs=[], outputs=[], js=SYNC_PLAY_JS, queue=False, show_api=False)
        sync_seek.click(fn=None, inputs=[], outputs=[], js=SYNC_SEEK_JS, queue=False, show_api=False)
        reset_videos.click(fn=None, inputs=[], outputs=[], js=RESET_VIDEOS_JS, queue=False, show_api=False)
        playback_speed.change(
            fn=None,
            inputs=[playback_speed],
            outputs=[],
            js=SET_SPEED_JS,
            queue=False,
            show_api=False,
        )
        refresh.click(
            refresh_history,
            inputs=[status_filter, source_filter],
            outputs=[jobs_table, job_selector, selected_job],
        )
        job_selector.change(fill_selected_job, inputs=[job_selector], outputs=[selected_job])
        load_job.click(
            inspect_job,
            inputs=[selected_job],
            outputs=[
                history_summary,
                history_status,
                history_postprocess_status,
                history_quality_summary,
                job_detail,
                history_logs,
                history_video,
                history_optimized_video,
                history_motion_download,
                history_video_download,
                history_optimized_motion_download,
                history_optimized_video_download,
                history_quality_download,
                history_zip_download,
            ],
        )
        postprocess_job.click(
            request_postprocess_selected,
            inputs=[selected_job, postprocess_profile, postprocess_pipeline, postprocess_render],
            outputs=[
                history_summary,
                history_status,
                history_postprocess_status,
                history_quality_summary,
                job_detail,
                history_logs,
                history_video,
                history_optimized_video,
                history_motion_download,
                history_video_download,
                history_optimized_motion_download,
                history_optimized_video_download,
                history_quality_download,
                history_zip_download,
            ],
        )
        retry_job.click(
            retry_selected,
            inputs=[selected_job],
            outputs=[
                history_summary,
                history_status,
                history_postprocess_status,
                history_quality_summary,
                job_detail,
                history_logs,
                history_video,
                history_optimized_video,
                history_motion_download,
                history_video_download,
                history_optimized_motion_download,
                history_optimized_video_download,
                history_quality_download,
                history_zip_download,
            ],
        )
        cancel_job.click(
            cancel_selected,
            inputs=[selected_job],
            outputs=[
                history_summary,
                history_status,
                history_postprocess_status,
                history_quality_summary,
                job_detail,
                history_logs,
                history_video,
                history_optimized_video,
                history_motion_download,
                history_video_download,
                history_optimized_motion_download,
                history_optimized_video_download,
                history_quality_download,
                history_zip_download,
            ],
        )

    return app.queue()
