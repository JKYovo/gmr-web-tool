import argparse
import json
from pathlib import Path

from general_motion_retargeting.stability_metrics import (
    analyze_motion_file,
    write_report,
)


def print_summary(summary):
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze robot motion COM, support polygon, and torso/waist lean."
    )
    parser.add_argument("--robot", default="elf3")
    parser.add_argument("--motion_path", required=True)
    parser.add_argument(
        "--root_rot_format",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="robot_motion.pkl normally stores xyzw. Some legacy BVH outputs used wxyz.",
    )
    parser.add_argument(
        "--support_height",
        type=float,
        default=0.08,
        help="Foot site height threshold in meters for support-foot detection.",
    )
    parser.add_argument("--json_path", default=None)
    parser.add_argument("--csv_path", default=None)
    args = parser.parse_args()

    report = analyze_motion_file(
        args.robot,
        args.motion_path,
        root_rot_format=args.root_rot_format,
        support_height=args.support_height,
    )

    json_path = args.json_path
    csv_path = args.csv_path
    if json_path is None and csv_path is None:
        motion_path = Path(args.motion_path)
        json_path = motion_path.with_name(motion_path.stem + "_stability.json")
        csv_path = motion_path.with_name(motion_path.stem + "_stability.csv")

    write_report(report, json_path=json_path, csv_path=csv_path)
    print_summary(report["summary"])
    if json_path:
        print(f"Saved JSON report: {json_path}")
    if csv_path:
        print(f"Saved CSV report: {csv_path}")
