#!/usr/bin/env python3
"""Compatibility wrapper for the old optimize_robot_motion.py command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def output_name_label(profile: str, pipeline: str) -> str:
    if pipeline == "v2_foot":
        return "foot" if profile == "soft" else f"{profile}_foot"
    if pipeline == "legacy":
        return f"{profile}_legacy"
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper. Prefer GMR/tools/motion_postprocess.py optimize."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot", default="elf3")
    parser.add_argument("--profile", default="preview", choices=["preview", "soft", "strict"])
    parser.add_argument("--pipeline", default="v2", choices=["legacy", "v2", "v2_foot"])
    parser.add_argument("--quality-json", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video-output", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # 保留旧脚本的命令行参数形式，实际处理逻辑统一交给新的通用后处理工具。
    tool = Path(__file__).with_name("motion_postprocess.py")
    output_path = Path(args.output).expanduser()
    label = output_name_label(args.profile, args.pipeline)
    quality_json = args.quality_json or str(output_path.with_name(f"quality_{label}.json"))
    video_output = args.video_output or str(output_path.with_name(f"preview_{label}.mp4"))
    cmd = [
        sys.executable,
        str(tool),
        "optimize",
        "--input",
        args.input,
        "--output",
        args.output,
        "--robot",
        args.robot,
        "--profile",
        args.profile,
        "--pipeline",
        args.pipeline,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--quality-json",
        quality_json,
    ]
    if args.render:
        cmd.append("--render")
        cmd.extend(["--video-output", video_output])

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
