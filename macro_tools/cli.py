"""Command-line orchestration for the three independent macro-context studies.

--audit-only reads annotation CSVs and builds complete-trial splits only.
--prepare-only also reads recordings, prepares inputs, and saves a short summary.
Without either flag, train and report final test metrics/plots. All settings and
outputs live in macro; source repositories and dataset remain read-only.
"""
import argparse
from collections import Counter
import copy
import sys
from macro_tools.paths import PROJECT_ROOT, create_run_directory, ensure_run_path, read_config, resolve_project_path, write_json
from macro_tools.sources import source_versions
from macro_tools.run_output import preparation_summary, save_run_error, update_run_config


def _config_path(value, category):
    if value.endswith(".json") or "/" in value or "\\" in value:
        return value
    return "configs/" + category + "/" + value + ".json"


def train(modality, *, config_path=None, experiment_path="step4_to_step4", protocol_path="protocol2",
          prepare_only=False, audit_only=False, run_name=None, **overrides):
    config = copy.deepcopy(read_config(config_path or "configs/" + modality + ".json"))
    if config.get("modality") != modality:
        raise ValueError("Configuration modality differs from entry point")
    experiment = read_config(_config_path(experiment_path, "experiments"))
    if overrides.get("steps123_percent") is not None:
        experiment["steps123_train_percent"] = overrides["steps123_percent"]
    from macro_tools.splits import validate_steps123_percent
    validate_steps123_percent(experiment)
    protocol = read_config(_config_path(protocol_path, "protocols"))
    if protocol.get("type") == "grouped_percentages":
        from macro_tools.splits import validate_percentage_protocol
        validate_percentage_protocol(protocol, config.get("training", {}).get("seed", 42))
    dataset = read_config(config.get("dataset_config", "configs/dataset.json"))
    if dataset.get("read_only") is not True:
        raise ValueError("dataset.read_only must remain true")
    for argument, key in (("epochs", "epochs"), ("batch_size", "batch_size"), ("device", "device"), ("workers", "num_workers"), ("amp", "amp")):
        if overrides.get(argument) is not None:
            config.setdefault("training", {})[key] = overrides[argument]
    for argument, key in (("points", "num_points"), ("frames", "num_frames"), ("image_size", "image_size")):
        if overrides.get(argument) is not None:
            config.setdefault("input", {})[key] = overrides[argument]
    # Only architecture settings change; retain the macro training/data setup.
    if modality in ("imu", "semg"):
        if overrides.get("model") is not None or str(config.get("model", "")).lower() == "tcn":
            from macro_tools.sources import activate_source
            activate_source(modality)
            from scripts.model_selection import select_model
            config = select_model(config, modality, overrides.get("model"))
    elif overrides.get("model") is not None:
        config["model"] = overrides["model"]
    for argument, key in (("sensors", "sensor_ids"), ("grids", "grid_ids"), ("views", "views"), ("subjects", "subject_ids")):
        if overrides.get(argument) is not None:
            config[key] = overrides[argument]
    from macro_tools.subject_splits import SUBJECT_PROTOCOL_TYPES, normalize_subject_protocol
    subject_arguments = ("test_subject", "validation_subject", "test_subjects", "validation_subjects",
                         "train_subject_count", "validation_subject_count", "test_subject_count")
    if any(overrides.get(key) is not None for key in subject_arguments) and protocol.get("type") not in SUBJECT_PROTOCOL_TYPES:
        raise ValueError("Subject holdout overrides apply only to protocol1")
    if protocol.get("type") in SUBJECT_PROTOCOL_TYPES:
        for part in ("test", "validation"):
            selected = overrides.get(part + "_subjects")
            singular = overrides.get(part + "_subject")
            if selected is not None and singular is not None:
                raise ValueError("Use either --{}-subject or --{}-subjects".format(part, part))
            if singular is not None:
                selected = [singular]
            if selected is not None:
                protocol.pop(part + "_subject_id", None)
                protocol[part + "_subject_ids"] = selected
                protocol.setdefault("subject_counts", {})[part] = len(selected)
                if part == "validation":
                    protocol.pop("validation_subject_id", None)
                    protocol.pop("validation", None)
        for part in ("train", "validation", "test"):
            count = overrides.get(part + "_subject_count")
            if count is not None:
                protocol.setdefault("subject_counts", {})[part] = count
                if part == "validation" and not (protocol.get("validation") or {}).get("subject_id"):
                    protocol.pop("validation", None)
        protocol = normalize_subject_protocol(protocol)
    run = create_run_directory(modality, config, experiment, protocol, run_name, prepare_only, audit_only)
    write_json(run / "config.json", {"modality": modality, "config": config, "dataset": dataset, "experiment": experiment, "protocol": protocol})
    update_run_config(run, status="preparing", synthetic=False)
    print("Preparing {} | experiment={} | protocol={} | model={}".format(modality.upper(), experiment["name"], protocol["name"], config["model"]), flush=True)
    runtime_dataset = dict(dataset, dataset_root=str(resolve_project_path(dataset["dataset_root"])))
    try:
        from macro_tools.annotations import load_annotations
        from macro_tools.splits import build_splits
        records = load_annotations(runtime_dataset, modality, config.get("subject_ids"))
        folds = build_splits(records, experiment, protocol, seed=config.get("training", {}).get("seed", 42))
        issues = [dict(issue, fold=fold["name"]) for fold in folds for issue in fold.get("issues", [])]
        update_run_config(run, preparation=preparation_summary(folds, issues))
        for fold in folds:
            print("{}: train={}, validation={}, test={} annotated micro gestures".format(fold["name"], *(len(fold[p]) for p in ("train", "validation", "test"))), flush=True)
            if "subject_counts" in fold.get("metadata", {}):
                meta = fold["metadata"]
                print("{}: participants train/validation/test {}/{}/{} | train [{}] | validation [{}] | test [{}]".format(
                    fold["name"], *(meta["subject_counts"][p] for p in ("train", "validation", "test")),
                    *(", ".join(meta[p + "_subject_ids"]) or "none" for p in ("train", "validation", "test"))), flush=True)
                if meta["excluded_subject_ids"]:
                    print("Unused participants: " + ", ".join(meta["excluded_subject_ids"]), flush=True)
            macro_split = fold.get("metadata", {}).get("macro_percentage_split")
            if macro_split:
                print("{}: step4 assembly split train/validation/test {}/{}/{} | requested {:g}/{:g}/{:g}% | actual annotations {:.2f}/{:.2f}/{:.2f}%".format(
                    fold["name"], *(macro_split["assembly_counts"][p] for p in ("train", "validation", "test")),
                    *(macro_split["requested_percentages"][p] for p in ("train", "validation", "test")),
                    *(macro_split["actual_annotation_percentages"][p] for p in ("train", "validation", "test"))), flush=True)
                if not fold["metadata"]["step4_train_validation_used"]:
                    print("Step4 train/validation groups are unused in source-only training; validation uses steps1-3.", flush=True)
            sampling = fold.get("metadata", {}).get("steps123_training_sampling")
            if sampling:
                print("{}: steps1-3 training uses {}/{} annotations ({:.1f}% actual; {:g}% requested), {}/{} whole trials".format(
                    fold["name"], sampling["selected_records"], sampling["eligible_records"],
                    sampling["actual_record_percent"], sampling["requested_percent"],
                    sampling["selected_trial_count"], sampling["eligible_trials"]), flush=True)
        if audit_only:
            update_run_config(run, status="audited")
            return run
        from macro_tools.adapters import prepare_dataset
        prepared = prepare_dataset(records, folds, runtime_dataset, config)
        issues = prepared.get("issues", issues)
        update_run_config(run, preparation=preparation_summary(prepared.get("folds", []), issues, prepared["label_mapping"]))
        counts = Counter(issue.get("code", "unknown") for issue in issues if issue.get("severity") == "warning")
        if counts:
            print("Preparation warnings: " + ", ".join("{}={}".format(k,v) for k,v in sorted(counts.items())), flush=True)
        if prepared.get("valid", True) is False or not prepared.get("folds"):
            raise ValueError("Preparation did not produce valid nonempty folds")
        from macro_tools.training import save_prepared_fold, train_fold
        results = []
        for index, fold in enumerate(prepared["folds"], start=1):
            if prepare_only and not config["training"].get("save_windows", False):
                continue
            output = ensure_run_path(run / fold["name"])
            if output.parent != run:
                raise ValueError("Invalid fold name")
            output.mkdir(exist_ok=False)
            save_prepared_fold(fold, output, config["training"].get("save_windows", False))
            if not prepare_only:
                print("Training fold {}/{}: {} (fresh model)".format(index, len(prepared["folds"]), fold["name"]), flush=True)
                update_run_config(run, status="training", current_fold=fold["name"])
                results.append(train_fold(fold, prepared["label_mapping"], config, output, runtime_dataset["dataset_root"]))
                write_json(run / "results.json", results)
        if results:
            from macro_tools.metrics import aggregate_fold_metrics, format_fold_summary
            from macro_tools.plots import save_training_report
            aggregate = aggregate_fold_metrics([r["test_metrics"] for r in results])
            write_json(run / "aggregate_metrics.json", aggregate)
            summary = format_fold_summary(results, aggregate)
            ensure_run_path(run / "fold_summary.txt").write_text(summary + "\n", encoding="utf-8")
            print(summary, flush=True)
            save_training_report(run, results=results)
        update_run_config(run, status="prepared" if prepare_only else "complete", current_fold=None)
        return run
    except Exception as error:
        save_run_error(run, error)
        print("Review this run: " + str(run.relative_to(PROJECT_ROOT)), file=sys.stderr)
        raise
    finally:
        update_run_config(run, source_versions=source_versions())


def main(modality, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--experiment", dest="experiment_path", default="step4_to_step4", help="Experiment short name or JSON path")
    parser.add_argument("--protocol", dest="protocol_path", default="protocol2", help="protocol1, protocol2, protocol2_percentages, or JSON path")
    parser.add_argument("--steps123-percent", type=float,
                        help="Percentage of eligible steps1-3 training trials to retain (0-100; default 100). Held-out sets stay fixed.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    parser.add_argument("--subjects", nargs="+")
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--test-subject", help="Protocol1: one fixed test participant (single fold)")
    test_group.add_argument("--test-subjects", nargs="+", help="Protocol1: fixed test participants, e.g. 1Alice 2Bingqian; one fold")
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument("--validation-subject", help="Protocol1: one fixed validation participant")
    validation_group.add_argument("--validation-subjects", nargs="+", help="Protocol1: fixed validation participants (requires fixed test participants)")
    parser.add_argument("--train-subject-count", type=int, help="Protocol1: number of training participants; default all remaining")
    parser.add_argument("--validation-subject-count", type=int, help="Protocol1: number of validation participants (default 1; 0 disables validation)")
    parser.add_argument("--test-subject-count", type=int, help="Protocol1: 1 runs every participant as test unless fixed IDs are supplied; larger counts require --test-subjects")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable CUDA FP16 mixed precision; --no-amp uses FP32 on the selected device (sEMG default)")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run-name")
    if modality == "imu":
        parser.add_argument("--sensors", nargs="+", choices=("E45A", "E477"))
    elif modality == "semg":
        parser.add_argument("--grids", nargs="+", choices=("grid1", "grid2"))
    if modality in ("imu", "semg"):
        transformer = "If_ConvTransformer_W" if modality == "imu" else "VisionTransformer"
        parser.add_argument("--model", choices=("tcn", "TCN", "transformer", transformer),
                            help="Signal architecture; switching resets only model_parameters")
        parser.add_argument("--points", type=int)
    else:
        parser.add_argument("--model", choices=("tsm", "mvtn"))
        parser.add_argument("--views", nargs="+", choices=("above", "front", "left", "right"))
        parser.add_argument("--frames", type=int)
        parser.add_argument("--image-size", type=int)
    try:
        run = train(modality, **vars(parser.parse_args(argv)))
        print("Saved run: " + str(run.relative_to(PROJECT_ROOT)), flush=True)
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print("Macro {} pipeline stopped: {}".format(modality, error), file=sys.stderr)
        return 2
