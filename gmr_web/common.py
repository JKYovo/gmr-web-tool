import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IK_CONFIG_ROOT = PROJECT_ROOT / "general_motion_retargeting" / "ik_configs"
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
JOB_ROOT = RUNTIME_ROOT / "jobs"
DB_PATH = RUNTIME_ROOT / "db" / "gmr_job_db.sqlite"
DEFAULT_HOST = os.environ.get("GMR_WEB_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("GMR_WEB_PORT", "7870"))

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
SUPPORTED_ROBOTS = ("elf3",)
SOURCE_REGISTRY = {
    "gvhmr_smplx": {
        "label": "GVHMR hmr4d_results.pt",
        "extensions": (".pt",),
        "ik_config": "smplx_to_elf3.json",
        "upload_enabled": True,
        "recommended": True,
    },
    "smplx_npz": {
        "label": "SMPL-X .npz",
        "extensions": (".npz",),
        "ik_config": "smplx_to_elf3.json",
        "upload_enabled": True,
        "recommended": False,
    },
    "bvh_xsens": {
        "label": "BVH Xsens / 3DSM",
        "extensions": (".bvh",),
        "ik_config": "bvh_xsens_to_elf3.json",
        "upload_enabled": True,
        "recommended": True,
    },
    "bvh_lafan1": {
        "label": "BVH Lafan1",
        "extensions": (".bvh",),
        "ik_config": "bvh_lafan1_to_elf3.json",
        "upload_enabled": True,
        "recommended": False,
    },
    "bvh_nokov": {
        "label": "BVH Nokov",
        "extensions": (".bvh",),
        "ik_config": "bvh_nokov_to_elf3.json",
        "upload_enabled": True,
        "recommended": False,
    },
    "fbx_offline": {
        "label": "FBX offline motion frames .pkl",
        "extensions": (".pkl",),
        "ik_config": "fbx_offline_to_elf3.json",
        "upload_enabled": True,
        "recommended": False,
    },
    "fbx": {
        "label": "FBX realtime / OptiTrack",
        "extensions": (),
        "ik_config": "fbx_to_elf3.json",
        "upload_enabled": False,
        "recommended": False,
    },
    "xrobot": {
        "label": "XRobot realtime",
        "extensions": (),
        "ik_config": "xrobot_to_elf3.json",
        "upload_enabled": False,
        "recommended": False,
    },
    "xsens_mvn": {
        "label": "Xsens MVN realtime",
        "extensions": (),
        "ik_config": "xsens_mvn_to_elf3.json",
        "upload_enabled": False,
        "recommended": False,
    },
}
SOURCE_TYPES = ("auto",) + tuple(SOURCE_REGISTRY.keys())
UPLOAD_SOURCE_TYPES = ("auto",) + tuple(
    name for name, spec in SOURCE_REGISTRY.items() if spec["upload_enabled"]
)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_job_id():
    return f"job_{uuid.uuid4().hex[:12]}"


def short_id(job_id):
    return job_id.replace("job_", "")[:8]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_source_type(input_path, source_type):
    if source_type != "auto":
        if source_type not in SOURCE_REGISTRY:
            raise ValueError(f"Unsupported source type: {source_type}")
        if not SOURCE_REGISTRY[source_type]["upload_enabled"]:
            raise ValueError(f"{source_type} is configured for ELF3 but is not a file-upload source yet.")
        return source_type
    suffix = Path(input_path).suffix.lower()
    if suffix == ".pt":
        return "gvhmr_smplx"
    if suffix == ".npz":
        return "smplx_npz"
    if suffix == ".bvh":
        return "bvh_xsens"
    if suffix == ".pkl":
        return "fbx_offline"
    raise ValueError(f"Cannot infer source type from suffix: {suffix}")


def validate_source_file(input_path, source_type):
    suffix = Path(input_path).suffix.lower()
    spec = SOURCE_REGISTRY[source_type]
    if spec["extensions"] and suffix not in spec["extensions"]:
        expected = ", ".join(spec["extensions"])
        raise ValueError(f"{source_type} expects file extension: {expected}")
    return spec


def source_config_path(source_type):
    return IK_CONFIG_ROOT / SOURCE_REGISTRY[source_type]["ik_config"]


def safe_stem(path):
    stem = Path(path).stem.strip() or "motion"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_input(input_file, output_dir):
    input_file = Path(input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if input_file.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_file}")
    staged = Path(output_dir) / f"input{input_file.suffix.lower()}"
    shutil.copy2(input_file, staged)
    return staged


def zip_artifacts(output_dir, artifact_paths):
    output_dir = Path(output_dir)
    zip_path = output_dir / "artifacts.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for artifact in artifact_paths:
            path = Path(artifact)
            if path.exists():
                zf.write(path, arcname=path.name)
    return zip_path
