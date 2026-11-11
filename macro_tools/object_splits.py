"""Protocol 3: configured object holdout within each step4 gesture.

Assign complete assemblies first, then retain training objects in training and
validation assemblies and held-out objects in test assemblies. Other rows are
explicitly excluded. The same held-out pairs are removed from steps1--3 inputs
in transfer experiments. This is gesture-only classification, with one fixed
object selection, not leave-one-object-out cross-validation or joint labels.

All operations are in memory. Raw recordings and annotations are never changed.
"""
from collections import Counter, defaultdict
from copy import deepcopy

from macro_tools.protocol_config import normalize_protocol


def validate_object_protocol(protocol):
    """Validate explicit object choices and a nested protocol-2 assembly plan."""
    if protocol.get("split_before_windowing") is not True:
        raise ValueError("Protocol3 requires split_before_windowing=true")
    choices = protocol.get("test_object_ids")
    if not isinstance(choices, dict) or not choices:
        raise ValueError("Protocol3 test_object_ids must map each gesture to object ID lists")
    for gesture, objects in choices.items():
        if (not isinstance(gesture, str) or not gesture.strip()
                or not isinstance(objects, list) or not objects
                or any(not isinstance(obj, str) or not obj.strip() or obj != obj.strip() for obj in objects)
                or len(objects) != len(set(objects))):
            raise ValueError("test_object_ids requires nonempty lists of unique string object IDs")
    assembly = protocol.get("assembly_split")
    if not isinstance(assembly, dict):
        raise ValueError("Protocol3 requires an assembly_split configuration")
    assembly = normalize_protocol(assembly)
    if assembly.get("type") not in ("held_out_repetitions", "held_out_macro_repetitions", "grouped_percentages"):
        raise ValueError("assembly_split must use held_out_repetitions or grouped_percentages")
    if assembly.get("split_before_windowing") is not True:
        raise ValueError("assembly_split requires split_before_windowing=true")
    source_validation = deepcopy(protocol.get("source_validation", {"strategy": "grouped_trials", "fraction": .2}))
    if not isinstance(source_validation, dict):
        raise ValueError("source_validation must be an object")
    return deepcopy(choices), assembly, source_validation


def build_object_split(original, eligible, target, labels, experiment, protocol, metadata, reasons, seed):
    """Build one fold; preserve the same step4 test records for all experiments."""
    from macro_tools.splits import (
        PARTS, build_splits, trial_key, _validation_pool, _apply_source_sampling, _finish,
    )
    choices, assembly, source_validation = validate_object_protocol(protocol)
    if set(choices) != set(labels):
        raise ValueError("test_object_ids must match target gestures; missing={}, unknown={}".format(
            sorted(set(labels) - set(choices)), sorted(set(choices) - set(labels))))
    available = defaultdict(set)
    for row in target:
        available[row["gesture_label"]].add(row["object_id"])
    for label in labels:
        held = set(choices[label])
        if not held.issubset(available[label]):
            raise ValueError("Unknown step4 test object IDs for {}: {}".format(label, sorted(held - available[label])))
        if not available[label] - held:
            raise ValueError("Protocol3 must retain at least one training object for " + label)

    # Reuse existing assembly rules, with a guaranteed non-recursive protocol type.
    base = build_splits(target, {"name": "step4_to_step4", "target_gesture_labels": labels},
                        assembly, seed=seed)[0]
    reasons = dict(reasons)
    for row in base["excluded"]:
        reasons[row["record_id"]] = "outside_assembly_split"

    def held(row):
        return row["object_id"] in choices[row["gesture_label"]]

    macro_parts = {}
    for part in PARTS:
        macro_parts[part] = [r for r in base[part] if held(r) == (part == "test")]
        for row in base[part]:
            if held(row) != (part == "test"):
                reasons[row["record_id"]] = ("training_object_in_test_assembly" if part == "test"
                                              else "held_out_object_in_training_assembly")
    use_macro = "step4" in metadata["train_sessions"]
    source = [r for r in eligible if r["session_id"] in metadata["train_sessions"] and r["session_id"] != "step4"]
    for row in source:
        if held(row):
            reasons[row["record_id"]] = "held_out_object_in_source"
    source = [r for r in source if not held(r)]
    if use_macro:
        assigned = dict(train=macro_parts["train"] + source,
                        validation=macro_parts["validation"], test=macro_parts["test"])
        validation = deepcopy(base["metadata"]["validation"])
        validation_sessions = ["step4"]
    else:
        train, validation_rows, validation = _validation_pool(source, labels, source_validation, seed)
        assigned = dict(train=train, validation=validation_rows, test=macro_parts["test"])
        validation_sessions = list(metadata["train_sessions"])
        for part in ("train", "validation"):
            for row in base[part]:
                reasons[row["record_id"]] = "step4_not_used_for_source_only_training"

    # Fail explicitly if object filtering removed a class from an active split.
    for part in PARTS:
        expected_nonempty = part != "validation" or (
            bool(base["validation"]) if use_macro else source_validation.get("strategy") != "none")
        if expected_nonempty:
            missing = set(labels) - {r["gesture_label"] for r in assigned[part]}
            if missing:
                raise ValueError("Protocol3 {} lacks gestures after object filtering: {}".format(part, sorted(missing)))
    name = protocol.get("name", "protocol3")
    meta = dict(metadata, protocol=name, protocol_type="held_out_objects",
                object_holdout_scope="within_gesture", test_object_ids=choices,
                step4_train_object_ids={label: sorted(available[label] - set(choices[label])) for label in labels},
                assembly_split=deepcopy(assembly),
                assembly_counts_before_object_filter={p: len({trial_key(r) for r in base[p]}) for p in PARTS},
                step4_annotation_counts_before_object_filter={p: len(base[p]) for p in PARTS},
                step4_annotation_counts_after_object_filter={p: len(macro_parts[p]) for p in PARTS},
                step4_train_validation_used=use_macro,
                validation=validation, validation_sessions=validation_sessions,
                source_object_exclusion="same held-out gesture-object pairs as step4",
                participant_scope="participants may occur in multiple partitions; assembly groups are disjoint")
    if "macro_percentage_split" in base["metadata"]:
        # Stored nested: these percentages describe full assemblies before filtering.
        meta["assembly_percentage_split"] = base["metadata"]["macro_percentage_split"]
    reasons = _apply_source_sampling(assigned, meta, reasons, experiment, seed)
    result = _finish(name, original, assigned, meta, labels, reasons)
    result["metadata"]["excluded_reason_counts"] = dict(sorted(Counter(
        r["exclusion_reason"] for r in result["excluded"]).items()))
    return result
