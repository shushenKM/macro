"""Portable configuration and output paths for the independent macro project.

Every generated artifact must be below this repository's runs/. Dataset and
model repositories are sibling directories, resolved only at runtime.
"""
import hashlib
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def ensure_run_path(value):
    root = PROJECT_ROOT.resolve()
    boundary = (root / "runs").resolve()
    path = resolve_project_path(value)
    if not boundary.is_relative_to(root) or not path.is_relative_to(boundary):
        raise ValueError("Macro outputs must remain under macro/runs/")
    return path


def read_config(value):
    value = json.loads(resolve_project_path(value).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    return value


def write_json(path, value):
    path = ensure_run_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError("Cannot serialize " + type(item).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=convert) + "\n", encoding="utf-8")


def experiment_run_name(modality, config, protocol, *, prepare_only=False, synthetic=False):
    """Describe effective settings after CLI overrides; never use the clock.

    Epochs is the requested limit, even if early stopping finishes sooner.
    Full settings remain in config.json. Keep names short enough for Windows;
    unusually long names receive a content hash after truncation.
    """
    video = modality in ("rgb", "depth")
    training, inputs = config.get("training") or {}, config.get("input") or {}
    defaults = {"imu": "If_ConvTransformer_W", "semg": "VisionTransformer",
                "rgb": "tsm", "depth": "tsm"}
    def token(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-_") or "unknown"
    parts = [modality, config.get("model", defaults[modality]),
             "batch{}".format(training.get("batch_size", 4 if video else 64)),
             "epoch{}".format(training.get("epochs", 30)),
             protocol.get("name", protocol.get("type", "protocol"))]
    if protocol.get("type") == "grouped_percentages":
        parts.append("split" + "-".join("{:g}".format(protocol["split_percentages"][p]) for p in ("train", "validation", "test")))
        parts.append("ss{}".format(protocol.get("split_seed", training.get("seed", 42))))
    if protocol.get("type") in ("subject_holdout", "leave_one_subject_out"):
        from macro_tools.subject_splits import normalize_subject_protocol
        subject_protocol = normalize_subject_protocol(protocol)
        counts = subject_protocol["subject_counts"]
        parts.append("subjects" + "-".join(str(counts[p]) if counts[p] is not None else "auto"
                                          for p in ("train", "validation", "test")))
        parts.append("loso" if subject_protocol["subject_selection"] == "leave_one_subject_out"
                     else "test-" + "-".join(token(s) for s in subject_protocol["test_subject_ids"]))
        if subject_protocol["validation_subject_ids"]:
            parts.append("val-" + "-".join(token(s) for s in subject_protocol["validation_subject_ids"]))
    if "steps123_train_percent" in protocol:
        parts.append("src{:g}pct".format(protocol["steps123_train_percent"]))
    key = "views" if video else "sensor_ids" if modality == "imu" else "grid_ids"
    selection = config.get(key) or []
    if selection:
        parts.append("-".join(token(item) for item in selection))
    if video:
        parts.extend(["frames{}".format(inputs.get("num_frames", 8)),
                      "size{}".format(inputs.get("image_size", 128))])
    else:
        parts.append("points{}".format(inputs.get("num_points", 128)))
    parts.extend(["lr{}".format(training.get("learning_rate", .0001 if video else .0005)),
                  "seed{}".format(training.get("seed", 42))])
    if protocol.get("type") not in ("subject_holdout", "leave_one_subject_out"):
        for key, prefix in (("test_subject_id", "test"), ("validation_subject_id", "val")):
            if protocol.get(key):
                parts.append(prefix + "-" + token(protocol[key]))
    if synthetic:
        parts.append("synthetic")
    if prepare_only:
        parts.append("prepare")
    name = "_".join(token(part) for part in parts)
    if len(name) > 160:
        name = name[:147].rstrip("._-") + "_" + hashlib.sha256(name.encode()).hexdigest()[:12]
    return name


def create_run_directory(modality, config, experiment, protocol, name=None, prepare_only=False, audit_only=False):
    if modality not in ("imu", "semg", "rgb", "depth"):
        raise ValueError("Unknown modality")
    automatic = name is None
    combined = dict(protocol, name=experiment["name"] + "_" + protocol["name"])
    if experiment["name"] in ("steps1234_to_step4", "steps123_to_step4"):
        combined["steps123_train_percent"] = experiment.get("steps123_train_percent", 100)
    if automatic:
        name = experiment_run_name(modality, config, combined, prepare_only=prepare_only)
        if audit_only:
            name += "_audit"
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Run name must be a single simple directory name")
    output = ensure_run_path(PROJECT_ROOT / "runs" / modality)
    output.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        path = ensure_run_path(output / (name if number == 1 else name + "_run%02d" % number))
        try:
            path.mkdir()
            return path
        except FileExistsError:
            if not automatic:
                raise
            number += 1
