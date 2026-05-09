import os
import shlex
import subprocess
from pathlib import Path


def external_backend_enabled():
    return os.environ.get("GMR_BACKEND", "internal").strip().lower() == "external"


def external_postprocess_available():
    return bool(os.environ.get("GMR_POSTPROCESS_CMD", "").strip())


def output_name_label(profile, pipeline):
    if pipeline == "v2_foot":
        return "foot" if profile == "soft" else f"{profile}_foot"
    if pipeline == "legacy":
        return f"{profile}_legacy"
    return profile


def default_optimize_output(input_path, profile, pipeline):
    input_path = Path(input_path)
    return input_path.with_name(f"motion_{output_name_label(profile, pipeline)}.pkl")


def default_optimize_quality_output(input_path, profile, pipeline):
    input_path = Path(input_path)
    return input_path.with_name(f"quality_{output_name_label(profile, pipeline)}.json")


def default_video_output(input_path, profile, pipeline):
    input_path = Path(input_path)
    return input_path.with_name(f"preview_{output_name_label(profile, pipeline)}.mp4")


def _command_from_env(env_name):
    command = os.environ.get(env_name, "").strip()
    if not command:
        raise RuntimeError(f"{env_name} is required when using the external GMR backend.")
    return shlex.split(command)


def _run_external_command(env_name, args, logger=None):
    cmd = _command_from_env(env_name) + list(args)
    if logger:
        logger(f"[External] Running: {' '.join(shlex.quote(part) for part in cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if logger:
            logger(f"[External] {line.rstrip()}")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"External command failed with exit code {return_code}: {cmd[0]}")


def run_external_retarget(job, logger=None):
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_external_command(
        "GMR_RETARGET_CMD",
        [
            "--input",
            job["input_file"],
            "--source-type",
            job["source_type"],
            "--robot",
            job["robot"],
            "--output-dir",
            str(output_dir),
            "--ground-clearance",
            str(job["ground_clearance"]),
            "--smoothing-alpha",
            str(job["smoothing_alpha"]),
            "--generate-video",
            "true" if job.get("generate_video", True) else "false",
        ],
        logger=logger,
    )

    motion_path = output_dir / "robot_motion.pkl"
    video_path = output_dir / "robot_preview.mp4"
    if not motion_path.exists():
        raise FileNotFoundError(f"External backend did not create {motion_path}")
    artifacts = {"motion_path": str(motion_path)}
    if video_path.exists():
        artifacts["video_path"] = str(video_path)
    return artifacts


def run_external_postprocess(job, logger=None):
    output_dir = Path(job["output_dir"])
    motion_path = Path(job.get("artifacts", {}).get("motion_path") or output_dir / "robot_motion.pkl")
    profile = job.get("postprocess_profile", "soft")
    pipeline = job.get("postprocess_pipeline", "v2_foot")
    render = bool(job.get("postprocess_render", True))

    if not motion_path.exists():
        raise FileNotFoundError(f"robot_motion.pkl is required before postprocess: {motion_path}")

    _run_external_command(
        "GMR_POSTPROCESS_CMD",
        [
            "--input",
            str(motion_path),
            "--output-dir",
            str(output_dir),
            "--robot",
            job["robot"],
            "--profile",
            profile,
            "--pipeline",
            pipeline,
            "--render",
            "true" if render else "false",
        ],
        logger=logger,
    )

    output_path = default_optimize_output(motion_path, profile, pipeline)
    quality_path = default_optimize_quality_output(motion_path, profile, pipeline)
    video_path = default_video_output(motion_path, profile, pipeline)

    if not output_path.exists():
        raise FileNotFoundError(f"External postprocess did not create {output_path}")

    artifacts = {"optimized_motion_path": str(output_path)}
    if quality_path.exists():
        artifacts["quality_report_path"] = str(quality_path)
    if video_path.exists():
        artifacts["optimized_video_path"] = str(video_path)
    return artifacts
