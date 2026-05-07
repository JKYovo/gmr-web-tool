import argparse
import pickle
import os
from pathlib import Path

import numpy as np


def convert_pkl_to_csv(pkl_path, out_folder):
    with pkl_path.open("rb") as f:
        motion_data = pickle.load(f)

    dof_pos = motion_data["dof_pos"]
    frame_rate = motion_data["fps"]
    motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
    motion[:, :3] = motion_data["root_pos"]
    motion[:, 3:7] = motion_data["root_rot"]
    motion[:, 7:] = dof_pos

    if frame_rate > 30:
        # downsample to 30 fps
        downsample_factor = frame_rate / 30.0
        indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
        old_length = motion.shape[0]
        motion = motion[indices]
        print(f"{pkl_path.name}: Downsampled from {old_length} to {motion.shape[0]} frames")

    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / pkl_path.with_suffix(".csv").name
    np.savetxt(out_path, motion, delimiter=",")
    print(f"Saved to {out_path}")
    return out_path


def resolve_input(args):
    provided = [value for value in (args.path, args.input, args.folder) if value]
    if len(provided) != 1:
        raise SystemExit("Please provide exactly one input: a positional path, --input, or --folder.")

    input_path = Path(provided[0]).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args.folder and not input_path.is_dir():
        raise ValueError(f"--folder expects a directory: {input_path}")
    if args.input and not input_path.is_file():
        raise ValueError(f"--input expects a .pkl file: {input_path}")
    if input_path.is_file() and input_path.suffix != ".pkl":
        raise ValueError(f"Input file must be a .pkl file: {input_path}")

    return input_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GMR pickle files to CSV (for beyondmimic)"
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a single GMR .pkl file or a folder containing .pkl files.",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a single GMR .pkl file.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to the folder containing pickle files from GMR.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="CSV output folder. Defaults to <input_folder>/csv or <pkl_parent>/csv.",
    )
    args = parser.parse_args()

    input_path = resolve_input(args)
    base_folder = input_path if input_path.is_dir() else input_path.parent
    out_folder = Path(args.output_folder).expanduser() if args.output_folder else base_folder / "csv"

    if input_path.is_file():
        convert_pkl_to_csv(input_path, out_folder)
    else:
        pkl_files = sorted(input_path.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {input_path}")
        for i, pkl_path in enumerate(pkl_files, start=1):
            print(f"({i}/{len(pkl_files)}) Converting {pkl_path.name}")
            convert_pkl_to_csv(pkl_path, out_folder)
