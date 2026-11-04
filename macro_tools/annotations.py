"""Read micro-gesture annotations for the separate macro-context experiments.

The dataset and sibling repositories are read-only. Existing annotation parsers
supply interval/clock handling; this adapter preserves composite step4 repetition
IDs such as 2_3, which the isolated-gesture parsers do not understand. With the
explicit 'prefix' policy, 2_3 means macro repetition 2, sub-action 3. All sub-actions
of the same subject/session/prefix, across all objects, receive one macro_trial_id.
The prefix identifies a complete assembly process, as confirmed by the user.
The suffix identifies a pick-and-install occurrence; source CSVs stay unchanged.

load_annotations(dataset_config, modality, subject_ids=None) returns parent
records before any split or signal processing. Raw repetition text and source
line numbers remain available. Output class labels stay micro-gesture labels.
"""
from copy import deepcopy
import csv
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERNS = {"imu_semg": "imu_semg_gesture_segments*_{locale}.csv",
                    "rgbd": "rgb_depth_gesture_segments*_{locale}.csv"}


def _name(value, description):
    if not isinstance(value, str) or not value or value in (".", "..") or any(c in value for c in ("/", "\\")):
        raise ValueError(description + " must be one folder/file name")
    return value


def repetition_metadata(record, raw, grouping="prefix"):
    """Return a copy with integer repetition and explicit compound-ID provenance."""
    result = deepcopy(record)
    value = str(raw or "").strip()
    step4 = result.get("session_id") == "step4"
    composite = re.fullmatch(r"([1-9][0-9]*)_([1-9][0-9]*)", value)
    numeric = re.fullmatch(r"([1-9][0-9]*)(?:\.0+)?", value)
    if composite and step4:
        if grouping != "prefix":
            raise ValueError("Composite step4 repetitions require explicit macro_repetition_grouping='prefix'")
        repetition, substep = map(int, composite.groups())
    elif numeric:
        repetition, substep = int(numeric.group(1)), None
    else:
        raise ValueError("Invalid repetition {!r} at {}:{}".format(value, result.get("source_file"), result.get("source_row")))
    result.update(repetition=repetition, repetition_raw=value,
                  macro_repetition=repetition if step4 else None,
                  macro_substep=substep, annotation_context="macro" if step4 else "isolated")
    result["parse_errors"] = [error for error in result.get("parse_errors", [])
                              if error.get("field") != "repetition"]
    group = "/".join(str(result[key]) for key in ("subject_id", "session_id", "object_id")) + "/" + str(repetition)
    if step4:
        group = "/".join(str(result[key]) for key in ("subject_id", "session_id")) + "/assembly" + str(repetition)
    result["trial_id"] = group
    result["macro_trial_id"] = group if step4 else None
    result["macro_grouping"] = grouping if step4 else None
    return result


def load_annotations(dataset_config, modality, subject_ids=None):
    """Read selected CSVs; add composite repetition metadata in memory only."""
    if modality not in ("imu", "semg", "rgb", "depth"):
        raise ValueError("Unknown modality: " + str(modality))
    if dataset_config.get("read_only") is not True:
        raise ValueError("dataset_config.read_only must remain true")
    root = Path(dataset_config.get("dataset_root", "../dataset"))
    root = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: " + str(root))
    locale = dataset_config.get("annotation_locale", "EN").upper()
    if locale not in ("EN", "FR") or (modality in ("rgb", "depth") and locale != "EN"):
        raise ValueError("Use EN annotations for RGB/depth; IMU/sEMG support EN or FR")
    folder_name = _name(dataset_config.get("folders", {}).get("annotations", "annotation"), "Annotation folder")
    family = "imu_semg" if modality in ("imu", "semg") else "rgbd"
    patterns = dataset_config.get("annotation_patterns") or DEFAULT_PATTERNS
    pattern = patterns.get(family, DEFAULT_PATTERNS[family]).replace("{locale}", locale)
    _name(pattern, "Annotation pattern")
    selected = subject_ids if subject_ids is not None else dataset_config.get("subject_ids")
    subjects = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / folder_name).is_dir()) if selected is None else list(selected)
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("Choose at least one unique subject")
    from macro_tools.sources import activate_source
    activate_source(modality)
    if family == "imu_semg":
        from scripts.annotations import read_annotations
    else:
        from scripts.rgbd_annotations import read_rgbd_annotations
    records = []
    for subject in sorted(subjects):
        _name(subject, "Subject ID")
        directory = (root / subject / folder_name).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("Annotation directory escapes the dataset")
        files = sorted(p for p in directory.glob(pattern) if p.is_file() and p.name.lower().endswith("_" + locale.lower() + ".csv"))
        if len(files) != 1:
            raise ValueError("Expected one {} for {}; found {}".format(pattern, subject, len(files)))
        path = files[0].resolve()
        if not path.is_relative_to(root):
            raise ValueError("Annotation file escapes the dataset")
        parsed = read_annotations(path, subject, locale) if family == "imu_semg" else read_rgbd_annotations(path, subject)
        raw_rows = {}
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter="," if locale == "EN" else ";")
            headers = [cell.strip() for cell in next(reader, [])]
            while True:
                line = reader.line_num + 1
                try:
                    cells = next(reader)
                except StopIteration:
                    break
                raw_rows[line] = dict(zip(headers, cells))
        for record in parsed:
            raw = raw_rows.get(record["source_row"], {}).get("repetition", "")
            # Only step4 introduces composite repetition strings; preserve all source text.
            record = repetition_metadata(record, raw, dataset_config.get("macro_repetition_grouping", "prefix"))
            records.append(record)
    return records
