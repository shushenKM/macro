"""Prepare macro experiments using read-only sibling readers and preprocessing.

Inputs are already split parent annotations. Only records assigned to training,
validation or test are loaded. Step4 uses one complete annotated micro-gesture
interval per example, including screw; steps1--3 retain their existing screw
sampling. Signal scaling is fitted on each fold's training examples only.
RGB/depth stay lazy: preparation records frame indices and depth paths, and the
existing decoder reads only requested samples during training. This module
performs no writes. All outputs are returned to the macro-owned trainer.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import math
import re
import numpy as np
from macro_tools.sources import activate_source

PARTITIONS = ("train", "validation", "test")
PROVENANCE = ("record_id", "subject_id", "session_id", "object_id", "repetition", "repetition_raw", "gesture_label", "source_file", "source_row", "macro_trial_id")


class PreparationError(ValueError):
    def __init__(self, issues, manifest=None):
        self.issues = issues
        self.manifest = manifest or []
        super().__init__("Preparation requires review: " + ", ".join(sorted({i["code"] for i in issues if i.get("severity") == "error"})))


def _issue(issues, code, message, record=None, severity="error", **detail):
    item = {key: record.get(key) for key in PROVENANCE} if record else {}
    item.update(detail, code=code, message=str(message), severity=severity)
    issues.append(item)


def _block(issues, manifest):
    if any(i.get("severity") == "error" for i in issues):
        raise PreparationError(issues, manifest)


def _inside(root, path):
    root = Path(root).resolve()
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Recording path escapes dataset root")
    return path


def resolve_imu_recording(root, subject, session, sensor, folder="imu", overrides=None):
    """Extend the source reader's filename resolver to step4 without modifying it."""
    match = re.fullmatch(r"(?:step|session)?([1-4])", str(session), re.IGNORECASE)
    if not match or sensor not in ("E45A", "E477"):
        raise ValueError("Expected step1--step4 and sensor E45A or E477")
    for component in (subject, folder):
        if not isinstance(component, str) or not component or component in (".", "..") or any(c in component for c in ("/", "\\", ":")):
            raise ValueError("Recording folder components must be simple names")
    number = match.group(1)
    key = "{}/step{}/{}".format(subject, number, sensor)
    if overrides and key in overrides:
        relative = Path(overrides[key])
        if relative.is_absolute():
            raise ValueError("Recording override must be dataset-relative")
        path = _inside(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    directory = _inside(root, Path(subject) / folder)
    pattern = re.compile(r"session{}_Shimmer_{}_Calibrated_PC.*\.csv".format(number, sensor), re.IGNORECASE)
    paths = sorted(p for p in directory.iterdir() if p.is_file() and pattern.fullmatch(p.name)) if directory.is_dir() else []
    if len(paths) != 1:
        raise ValueError("Expected one session{} {} IMU file for {}; found {}".format(number, sensor, subject, len(paths)))
    return _inside(root, paths[0])


def resolve_video_folder(root, dataset, subject, session, view):
    """Use an explicit portable camera-folder override for nonstandard names."""
    key = "/".join((subject, session, view))
    override = (dataset.get("video_recording_overrides") or {}).get(key)
    if override is not None:
        if Path(override).is_absolute():
            raise ValueError("Video recording overrides must be dataset-relative")
        return _inside(root, override)
    mapping = dataset.get("session_mapping") or {}
    if session not in mapping:
        raise ValueError("No recording session mapping for " + session)
    return _inside(root, Path(subject) / dataset.get("folders", {}).get("rgbd", "rgbd") / mapping[session] / view)


def _interval(record, key):
    interval = record.get("intervals", {}).get(key, {})
    start, end = interval.get("start_s"), interval.get("end_s")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
        raise ValueError("{} requires finite 0 <= start < end".format(key))
    return start, end


def _assigned(folds):
    selected = {}
    for fold in folds:
        for part in PARTITIONS:
            for record in fold[part]:
                selected.setdefault(record["record_id"], record)
    if not selected:
        raise ValueError("No assigned parent annotations")
    return list(selected.values())


def _provenance(record, index=0):
    item = {key: record.get(key) for key in PROVENANCE}
    item.update(parent_record_id=record["record_id"], window_id="{}:window{}".format(record["record_id"], index))
    return item


def _diagnostics(prepared, issues, record):
    for item in getattr(prepared, "diagnostics", []):
        _issue(issues, item.get("code", "signal_diagnostic"), item.get("message", str(item)), record,
               severity=item.get("severity", "warning"), diagnostic=item)


def _full_imu_example(record, prepared, sensors, points):
    blocks, bounds, support = [], {}, {}
    for sensor in sensors:
        start, end = _interval(record, "imu_" + sensor.lower())
        data = prepared[sensor]
        times, values = np.asarray(data.timestamps_s), np.asarray(data.features)
        selected = (times >= -1e-9) & (times < end - start - 1e-9)
        times, values = times[selected], values[selected]
        if len(times) < 2 or values.shape != (len(times), 9) or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
            raise ValueError("A full IMU interval needs at least two finite, increasing processed samples")
        grid = np.linspace(times[0], times[-1], points)
        blocks.append(np.stack([np.interp(grid, times, values[:, c]) for c in range(9)]))
        bounds[sensor] = dict(start_s=start, end_s=end)
        support[sensor] = dict(first_s=float(start+times[0]), last_s=float(start+times[-1]), processed_samples=len(times))
    metadata = _provenance(record)
    metadata.update(sampling="full_micro_gesture", time_axis="normalized_gesture_progress", num_points=points,
                    sensor_ids=list(sensors), sensor_windows=bounds, source_support=support)
    return [(np.concatenate(blocks, axis=0)[None].astype(np.float32), metadata)]


def _signal_examples(records, dataset, config, issues, manifest):
    modality = config["modality"]
    root = Path(dataset["dataset_root"]).resolve()
    quality = config.get("data_quality") or {}
    cache, windows = {}, {}
    if modality == "imu":
        from imu.dataset import _settings
        from imu.recordings import read_imu_recording
        from imu.preprocessing import preprocess_segment
        from imu.sampling import gesture_examples
        preprocessing, sensors, points, duration = _settings(config)
        shape = (1, 9 * len(sensors), points)
        axes = (0, 1, 3)
    else:
        from semg.dataset import _settings, _examples
        from semg.recordings import read_semg_recording, resolve_semg_recording
        from semg.preprocessing import preprocess_segment
        grids, points, layout, duration, count, policy = _settings(config)
        channels = [c-1 for grid in grids for c in layout[grid]]
        shape = (8, points, 8*len(grids))
        axes = (0, 2)
    for record in records:
        entry = _provenance(record)
        entry.update(status="pending", recordings={})
        manifest.append(entry)
        try:
            if modality == "imu":
                prepared = {}
                for sensor in sensors:
                    key = (record["subject_id"], record["session_id"], sensor)
                    if key not in cache:
                        path = resolve_imu_recording(root, *key, folder=dataset.get("folders", {}).get("imu", "imu"), overrides=quality.get("recording_overrides"))
                        cache[key] = read_imu_recording(path, sensor, acceleration=preprocessing.get("acceleration", "Accel_WR"))
                    recording = cache[key]
                    entry["recordings"][sensor] = recording.path.resolve().relative_to(root).as_posix()
                    prepared[sensor] = preprocess_segment(recording, *_interval(record, "imu_" + sensor.lower()), preprocessing)
                    _diagnostics(prepared[sensor], issues, record)
                if record["session_id"] == "step4":
                    values = _full_imu_example(record, prepared, sensors, points)
                else:
                    parent = deepcopy(record)
                    if parent["gesture_label"] == "screw":
                        lengths = []
                        for sensor in ("E45A", "E477"):
                            if "imu_" + sensor.lower() in record.get("intervals", {}):
                                start, end = _interval(record, "imu_" + sensor.lower())
                                lengths.append(end-start)
                        parent["screw_shared_parent_duration_s"] = min(lengths)
                    values = gesture_examples(parent, prepared, sensors, points, duration)
            else:
                key = (record["subject_id"], record["session_id"])
                if key not in cache:
                    path = resolve_semg_recording(root, *key, semg_folder=dataset.get("folders", {}).get("semg", "semg"), overrides=quality.get("recording_overrides"))
                    cache[key] = read_semg_recording(path)
                recording = cache[key]
                entry["recordings"]["semg"] = recording.path.resolve().relative_to(root).as_posix()
                start, end = _interval(record, "semg")
                prepared = preprocess_segment(recording, start, end, channels, config.get("preprocessing") or {})
                _diagnostics(prepared, issues, record)
                if record["session_id"] == "step4":
                    values = _examples(record, prepared, grids, layout, points, end-start, 1, "error")
                    for _, metadata in values:
                        metadata["sampling"] = "full_micro_gesture"
                else:
                    values = _examples(record, prepared, grids, layout, points, duration, count, policy)
            if not values:
                raise ValueError("Interval produced no model examples")
            for _, metadata in values:
                metadata.update({key: record.get(key) for key in PROVENANCE})
            windows[record["record_id"]] = values
            entry.update(status="prepared", window_count=len(values))
        except (OSError, ValueError) as error:
            entry.update(status="blocked", window_count=0)
            _issue(issues, getattr(error, "code", "signal_preparation_error"), error, record)
    _block(issues, manifest)
    return windows, shape, axes


def _signal_folds(records, folds, dataset, config, labels, issues, manifest):
    windows, shape, axes = _signal_examples(records, dataset, config, issues, manifest)
    prepared = []
    for fold in folds:
        splits = {}
        for part in PARTITIONS:
            arrays, targets, metadata = [], [], []
            for record in fold[part]:
                for array, detail in windows[record["record_id"]]:
                    arrays.append(array)
                    targets.append(labels[record["gesture_label"]])
                    metadata.append(detail)
            splits[part] = dict(x=np.stack(arrays) if arrays else np.empty((0, *shape), dtype=np.float32),
                                y=np.asarray(targets, dtype=np.int64), metadata=metadata)
        train = splits["train"]["x"].astype(np.float64)
        if not len(train):
            raise ValueError("Empty training partition")
        mean, std = train.mean(axis=axes), train.std(axis=axes)
        constant = std < 1e-8
        if constant.any():
            _issue(issues, "constant_training_features", "Constant features use unit scale after training centering", severity="warning", fold=fold["name"])
        std[constant] = 1.0
        mean_view, std_view = ((mean[None, None, :, None], std[None, None, :, None]) if config["modality"] == "imu"
                               else (mean[None, :, None, :], std[None, :, None, :]))
        for part in PARTITIONS:
            splits[part]["x"] = ((splits[part]["x"] - mean_view) / std_view).astype(np.float32)
            if not np.isfinite(splits[part]["x"]).all():
                raise ValueError("Nonfinite normalized model input")
        prepared.append(dict(name=fold["name"], splits=splits,
            normalization=dict(mean=mean.tolist(), std=std.tolist(), fit_partition="train", axes=list(axes)),
            split_summary=_summary(fold, splits), protocol_metadata=fold.get("metadata", {}),
            excluded_records=[r["record_id"] for r in fold.get("excluded", [])]))
    return prepared


def _summary(fold, splits):
    result = {}
    for part in PARTITIONS:
        records = fold[part]
        result[part] = dict(parents=len(records), examples=len(splits[part]["y"]) if isinstance(splits[part], dict) else len(splits[part]),
                           subjects=sorted({r["subject_id"] for r in records}),
                           gesture_counts=dict(Counter(r["gesture_label"] for r in records)),
                           session_counts=dict(Counter(r["session_id"] for r in records)))
    return result


def _video_folds(records, folds, dataset, config, labels, issues, manifest):
    from vision.dataset import video_settings, _screw_settings, _parent_clips, RGB_MEAN, RGB_STD
    from vision.recordings import inspect_recording, read_depth_frames
    settings, screw = video_settings(config), _screw_settings(config)
    root = Path(dataset["dataset_root"]).resolve()
    recordings, samples = {}, {}
    for subject, session, view in sorted({(r["subject_id"], r["session_id"], v) for r in records for v in settings["views"]}):
        try:
            folder = resolve_video_folder(root, dataset, subject, session, view)
            item = inspect_recording(folder, root, settings["modality"], probe_frames=True)
            recordings[(subject, session, view)] = item
            manifest.append(dict({k: v for k, v in item.items() if k != "depth_map"}, subject_id=subject, session_id=session, view=view))
            if item.get("missing_depth_frame_count", 0):
                _issue(issues, "missing_depth_frames", "Absent depth images exist; selected annotation images are checked separately", severity="warning", subject_id=subject, session_id=session, view=view, count=item["missing_depth_frame_count"])
        except (OSError, ValueError) as error:
            _issue(issues, "recording_unavailable", error, subject_id=subject, session_id=session, view=view)
    _block(issues, manifest)
    depth_probed = set()
    for record in records:
        try:
            for view in settings["views"]:
                _interval(record, view)
            plan = (screw[0], 1, "error") if record["session_id"] == "step4" else screw
            clips = _parent_clips(record, recordings, settings, plan, labels, issues)
            for clip in clips:
                clip.update({key: record.get(key) for key in PROVENANCE})
                if settings["modality"] == "depth":
                    for view in clip["views"]:
                        if view["video_path"] not in depth_probed:
                            read_depth_frames(root, [view["depth_paths"][0]], settings["image_size"])
                            depth_probed.add(view["video_path"])
            samples[record["record_id"]] = clips
        except (OSError, ValueError) as error:
            _issue(issues, "invalid_clip_source", error, record)
    _block(issues, manifest)
    normalization = (dict(method="fixed_imagenet", mean=list(RGB_MEAN), std=list(RGB_STD), fitted=False) if settings["modality"] == "rgb" else
                     dict(method="fixed_depth_range", minimum=settings["depth_min"], maximum=settings["depth_max"], units="raw_recorded_depth_values", zero_is_missing=True, fitted=False))
    prepared = []
    for fold in folds:
        splits = {part: [sample for record in fold[part] for sample in samples[record["record_id"]]] for part in PARTITIONS}
        prepared.append(dict(name=fold["name"], splits=splits, normalization=normalization, split_summary=_summary(fold, splits),
                             protocol_metadata=fold.get("metadata", {}), excluded_records=[r["record_id"] for r in fold.get("excluded", [])]))
    return prepared


def prepare_dataset(records, folds, dataset_config, config):
    """Prepare only split-assigned source intervals and return arrays/manifests."""
    if dataset_config.get("read_only") is not True:
        raise ValueError("dataset.read_only must remain true")
    modality = config["modality"]
    activate_source(modality)
    dataset = deepcopy(dataset_config)
    root = Path(dataset["dataset_root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    dataset["dataset_root"] = str(root.resolve())
    if not root.is_dir():
        raise FileNotFoundError(root)
    assigned = _assigned(folds)
    labels = {name: index for index, name in enumerate(sorted({r["gesture_label"] for r in assigned}))}
    expected = config.get("expected_gesture_labels")
    if expected is not None and set(expected) != set(labels):
        raise ValueError("Assigned gesture vocabulary differs from expected_gesture_labels")
    issues = [dict(issue, fold=fold["name"]) for fold in folds for issue in fold.get("issues", [])]
    manifest = []
    _block(issues, manifest)
    function = _signal_folds if modality in ("imu", "semg") else _video_folds
    prepared = function(assigned, folds, dataset, config, labels, issues, manifest)
    return dict(valid=True, folds=prepared, label_mapping=labels, issues=issues, manifest=manifest, dataset_root=root.resolve())


def make_dataset(split, dataset_root, config, training=False):
    """Wrap prepared arrays or portable video manifests in a torch Dataset."""
    activate_source(config["modality"])
    if config["modality"] in ("imu", "semg"):
        import torch
        from torch.utils.data import TensorDataset
        return TensorDataset(torch.from_numpy(np.ascontiguousarray(split["x"], dtype=np.float32)),
                             torch.from_numpy(np.asarray(split["y"], dtype=np.int64)))
    from vision.dataset import VideoGestureDataset
    return VideoGestureDataset(split, dataset_root, config, training=training)
