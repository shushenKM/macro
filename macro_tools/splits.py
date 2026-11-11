"""Create source-context experiments with participant or whole-trial holdouts.

The prediction target is a micro gesture or an observed step4 gesture-object
pair. Raw gesture labels remain available to preprocessing. Every test record is from
step4. The three experiment names control training sessions independently of the
participant/repetition protocol. All assignments happen before windowing.

Protocol1 supports configurable participant counts: leave-one-subject-out folds
or one fixed test participant group. It excludes test, validation and unused
participants from every training session. Protocol2 can hold out named macro repetitions or shuffle complete
assemblies into configurable percentage partitions, independent of repetition ID.
Protocol3 first assigns complete assemblies using a nested protocol2 plan, then
holds out configured objects within each gesture. The same held-out pairs are
excluded from isolated-context source data in transfer experiments.
Experiment 2 adds isolated-context data to experiment 1's training partition;
experiment 3 uses only isolated-context training/validation, retaining the same
step4 test population. By default, class vocabulary is the observed step4
vocabulary; source-only labels are explicitly listed as exclusions, never renamed.
Overlapping annotated time intervals are recorded as warnings under the user's
accepted policy. Reused record IDs and trial groups across partitions are errors.

steps123_train_percent retains a seeded percentage of eligible source training
trials AFTER validation/test assignment. Sampling keeps trials intact and is
stratified by participant, session and gesture composition. Positive percentages
retain at least one trial per stratum; rounding means actual counts can differ
from the requested percentage. Step4, validation and test records are unchanged.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import random
from macro_tools.subject_splits import SUBJECT_PROTOCOL_TYPES, participant_partitions
from macro_tools.protocol_config import normalize_protocol, percentage_settings
from macro_tools.tasks import label_of, validate_task, with_task_labels

PARTS = ("train", "validation", "test")
EXPERIMENTS = {
    "step4_to_step4": ["step4"],
    "steps1234_to_step4": ["step1", "step2", "step3", "step4"],
    "steps123_to_step4": ["step1", "step2", "step3"],
}


def trial_key(record):
    """Whole macro execution, or isolated source trial, independent of gesture."""
    if record.get("session_id") == "step4":
        return "/".join((str(record["subject_id"]), "step4", "assembly" + str(record["repetition"])))
    return "/".join(str(record[key]) for key in ("subject_id", "session_id", "object_id", "repetition"))


def _ids(values, name):
    if not isinstance(values, list) or not values or any(type(value) is not int or value < 1 for value in values) or len(values) != len(set(values)):
        raise ValueError(name + " must contain unique positive integers")
    return set(values)


def validate_steps123_percent(experiment):
    """Validate the percentage before reading recordings or creating a run."""
    percent = experiment.get("steps123_train_percent", 100)
    if isinstance(percent, bool) or not isinstance(percent, (int, float)) or not math.isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError("steps123_train_percent must be a finite percentage between 0 and 100")
    if experiment.get("name") == "steps123_to_step4" and percent == 0:
        raise ValueError("steps123_to_step4 requires steps123_train_percent greater than 0; otherwise training is empty")
    if experiment.get("name") == "step4_to_step4" and percent != 100:
        raise ValueError("steps123_train_percent applies only to steps1234_to_step4 and steps123_to_step4")
    return percent


def _sample_source_training(assigned, percent, seed):
    """Select whole source trials without changing either held-out partition."""
    source = [r for r in assigned["train"] if r["session_id"] != "step4"]
    groups = defaultdict(list)
    for record in source:
        groups[trial_key(record)].append(record)
    strata = defaultdict(list)
    for key, rows in groups.items():
        signature = tuple(sorted({label_of(r) for r in rows}))
        strata[(rows[0]["subject_id"], rows[0]["session_id"], signature)].append(key)
    selected = set()
    rng = random.Random(seed)
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        count = 0 if percent == 0 else max(1, int(math.floor(len(keys) * percent / 100 + .5)))
        selected.update(keys[:count])
    retained = [r for r in source if trial_key(r) in selected]
    dropped = {r["record_id"] for r in source if trial_key(r) not in selected}
    assigned["train"] = [r for r in assigned["train"] if r["record_id"] not in dropped]
    meta = dict(requested_percent=percent, seed=seed,
                unit="whole_source_trial", stratification=["subject_id", "session_id", "gesture_composition"],
                minimum_one_trial_per_stratum=percent > 0,
                eligible_trials=len(groups), selected_trial_count=len(selected), selected_trials=sorted(selected),
                eligible_records=len(source), selected_records=len(retained),
                actual_record_percent=100 * len(retained) / len(source) if source else 0.0,
                eligible_gesture_counts=dict(sorted(Counter(r["gesture_label"] for r in source).items())),
                selected_gesture_counts=dict(sorted(Counter(r["gesture_label"] for r in retained).items())))
    return meta, dropped


def _apply_source_sampling(assigned, metadata, reasons, experiment, seed):
    if experiment["name"] == "step4_to_step4":
        return reasons
    meta, dropped = _sample_source_training(assigned, validate_steps123_percent(experiment), seed)
    metadata["steps123_training_sampling"] = meta
    return dict(reasons, **{rid: "steps123_training_subsample" for rid in dropped})


def validate_percentage_protocol(protocol, seed=42):
    """Validate percentages of complete step4 assemblies, not individual rows."""
    protocol = normalize_protocol(protocol)
    conflicting = {"train_repetitions", "test_repetitions", "validation"} & set(protocol)
    if conflicting and protocol.get("grouped_percentages") is not True:
        raise ValueError("grouped_percentages uses split_percentages and source_validation, not " + ", ".join(sorted(conflicting)))
    percentages, split_seed = percentage_settings(protocol, seed)
    source_validation = protocol.get("source_validation", {"strategy": "grouped_trials", "fraction": .2})
    if not isinstance(source_validation, dict) or source_validation.get("strategy", "grouped_trials") not in ("grouped_trials", "none"):
        raise ValueError("source_validation must specify grouped_trials or none")
    if source_validation.get("strategy") != "none":
        fraction = source_validation.get("fraction", .2)
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("source_validation.fraction must be between zero and one")
    return dict(percentages), split_seed, dict(source_validation)


def _percentage_macro_partitions(target, percentages, seed):
    """Allocate whole assemblies by largest remainder; every positive part gets one.

    All participants' assemblies are pooled. Counts target assembly percentages;
    unequal assembly sizes can make annotation percentages differ. Stable sorted
    group keys make the same seed independent of row order and source filenames.
    """
    keys = sorted({trial_key(r) for r in target})
    active = [part for part in PARTS if percentages[part] > 0]
    if len(keys) < len(active):
        raise ValueError("Percentage split needs at least {} complete assemblies; found {}".format(len(active), len(keys)))
    quotas = {part: len(keys) * percentages[part] / 100 for part in PARTS}
    counts = {part: max(int(percentages[part] > 0), math.floor(quotas[part])) for part in PARTS}
    while sum(counts.values()) < len(keys):
        part = max(active, key=lambda p: quotas[p] - counts[p])
        counts[part] += 1
    while sum(counts.values()) > len(keys):
        part = max((p for p in active if counts[p] > 1), key=lambda p: counts[p] - quotas[p])
        counts[part] -= 1
    random.Random(seed).shuffle(keys)
    selected, cursor = {}, 0
    for part in PARTS:
        selected[part] = set(keys[cursor:cursor + counts[part]])
        cursor += counts[part]
    partitions = {part: [r for r in target if trial_key(r) in selected[part]] for part in PARTS}
    meta = dict(requested_percentages=percentages, split_seed=seed, unit="whole_macro_assembly",
                allocation="largest_remainder_minimum_one_per_nonzero_partition",
                scope="all_selected_participants_pooled", total_assemblies=len(keys),
                assembly_counts=counts, selected_trials={p: sorted(selected[p]) for p in PARTS},
                annotation_counts={p: len(partitions[p]) for p in PARTS},
                actual_assembly_percentages={p: 100 * counts[p] / len(keys) for p in PARTS},
                actual_annotation_percentages={p: 100 * len(partitions[p]) / len(target) for p in PARTS})
    return partitions, meta


def _validation_pool(records, labels, settings, seed):
    if settings.get("strategy", "grouped_trials") == "none":
        return list(records), [], {"strategy": "none"}
    if settings.get("strategy", "grouped_trials") not in ("grouped_trials", "auto"):
        raise ValueError("Repetition protocol validation must use grouped_trials or none")
    fraction = settings.get("fraction", .2)
    if isinstance(fraction, bool) or not isinstance(fraction, (int,float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    if not records:
        raise ValueError("No eligible training/validation trials")
    groups = defaultdict(list)
    for record in records:
        groups[trial_key(record)].append(record)
    keys = sorted(groups)
    classes = {key: {label_of(record) for record in groups[key]} for key in keys}
    choices = {label: {key for key in keys if label in classes[key]} for label in labels}
    scarce = [label for label, candidates in choices.items() if len(candidates) < 2]
    if scarce:
        raise ValueError("Training/validation class coverage needs two eligible complete trials per class: " + ", ".join(sorted(scarce)))
    order = keys[:]
    random.Random(settings.get("seed", seed)).shuffle(order)
    priority = {key: i for i,key in enumerate(order)}
    target = max(1, min(len(keys)-1, int(math.floor(len(keys)*fraction+.5))))
    visited = set()
    def viable(selected, key):
        return all(not candidate.issubset(selected | {key}) for candidate in choices.values())
    def cover(selected):
        state = frozenset(selected)
        if state in visited:
            return None
        visited.add(state)
        covered = set().union(*(classes[key] for key in selected)) if selected else set()
        missing = set(labels) - covered
        if not missing:
            return selected
        options = {label: [key for key in choices[label] if key not in selected and viable(selected,key)] for label in missing}
        label = min(missing, key=lambda x: (len(options[x]),x))
        for key in sorted(options[label], key=lambda x: (-len(classes[x] & missing),priority[x])):
            found = cover(selected | {key})
            if found is not None:
                return found
        return None
    selected = cover(set())
    if selected is None:
        raise ValueError("Cannot split complete trials while retaining all classes in training and validation")
    for key in order:
        if len(selected) >= target:
            break
        if key not in selected and viable(selected,key):
            selected.add(key)
    return ([r for r in records if trial_key(r) not in selected],
            [r for r in records if trial_key(r) in selected],
            dict(strategy="grouped_trials", fraction=fraction, seed=settings.get("seed",seed),
                 eligible_trials=len(keys), selected_trials=sorted(selected)))


def _overlap_warnings(fold):
    timelines = defaultdict(list)
    for partition in PARTS:
        for record in fold[partition]:
            for timeline, interval in record.get("intervals", {}).items():
                lo, hi = interval.get("start_s"), interval.get("end_s")
                if isinstance(lo, (int,float)) and isinstance(hi,(int,float)) and math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi:
                    key = (record["subject_id"],record["session_id"],timeline)
                    timelines[key].append((lo,hi,partition,record["record_id"]))
    warnings=[]
    for key, items in timelines.items():
        active=[]
        for lo,hi,partition,record_id in sorted(items):
            active=[item for item in active if item[1]>lo]
            for other in active:
                if other[2] != partition:
                    warnings.append(dict(code="overlapping_intervals",severity="warning",
                        message="Annotated intervals overlap across partitions; retained under accepted overlap policy",
                        subject_id=key[0],session_id=key[1],timeline=key[2],
                        record_ids=[other[3],record_id],partitions=[other[2],partition]))
            active.append((lo,hi,partition,record_id))
    return warnings


def _finish(name, original, assigned, metadata, labels, reasons):
    fold = dict(name=name, **assigned)
    selected={record["record_id"] for part in PARTS for record in fold[part]}
    fold["excluded"]=[dict(record, exclusion_reason=reasons.get(record["record_id"],"outside_experiment_or_holdout"))
                      for record in original if record["record_id"] not in selected]
    fold["metadata"]=metadata
    fold["issues"]=[]
    seen_ids,seen_groups={},{}
    for part in PARTS:
        for record in fold[part]:
            rid=record["record_id"]
            if rid in seen_ids:
                raise ValueError("Source record assigned more than once: " + rid)
            seen_ids[rid]=part
            group=trial_key(record)
            if group in seen_groups and seen_groups[group]!=part:
                raise ValueError("Whole source trial crosses partitions: " + group)
            seen_groups[group]=part
        if part in ("train","test") and not fold[part]:
            raise ValueError(part + " partition is empty")
        missing=set(labels)-{label_of(record) for record in fold[part]}
        if missing and part=="train":
            raise ValueError("Training is missing target classes: " + ", ".join(sorted(missing)))
        if missing and fold[part] and part in ("validation","test"):
            if metadata.get("task") == "gesture_object":
                raise ValueError(part + " is missing target classes: " + ", ".join(sorted(missing)))
            fold["issues"].append(dict(code="missing_"+part+"_classes",severity="warning",
                message="The partition has no examples for some fixed target classes",missing_classes=sorted(missing)))
    if not fold["validation"]:
        fold["issues"].append(dict(code="no_validation_split",severity="warning",message="Last epoch will be used; no validation tuning"))
    fold["issues"].extend(_overlap_warnings(fold))
    fold["summary"]={part:dict(segments=len(fold[part]), trials=len({trial_key(r) for r in fold[part]}),
        subjects=sorted({r["subject_id"] for r in fold[part]}), sessions=sorted({r["session_id"] for r in fold[part]}),
        gesture_counts=dict(sorted(Counter(r["gesture_label"] for r in fold[part]).items())),
        class_counts=dict(sorted(Counter(label_of(r) for r in fold[part]).items()))) for part in PARTS+("excluded",)}
    fold["valid"]=True
    return fold


def build_splits(records, experiment, protocol, seed=42, task="gesture"):
    """Return macro-context folds; input records/configuration remain unchanged."""
    protocol = normalize_protocol(protocol)
    validate_task(task, protocol)
    records=with_task_labels(deepcopy(list(records)), task, protocol)
    if not records:
        raise ValueError("No annotation records")
    ids=[r["record_id"] for r in records]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate source record IDs")
    experiment={"name":experiment} if isinstance(experiment,str) else deepcopy(experiment)
    name=experiment.get("name")
    if name not in EXPERIMENTS:
        raise ValueError("Unknown macro experiment: " + str(name))
    validate_steps123_percent(experiment)
    percentage_settings = validate_percentage_protocol(protocol, seed) if protocol.get("type") == "grouped_percentages" else None
    train_sessions=experiment.get("train_sessions",EXPERIMENTS[name])
    test_sessions=experiment.get("test_sessions",["step4"])
    if set(test_sessions)!={"step4"} or set(train_sessions)!=set(EXPERIMENTS[name]):
        raise ValueError("Experiment session selections must match the named macro experiment")
    target=[r for r in records if r["session_id"]=="step4"]
    if not target:
        raise ValueError("No step4 annotations selected")
    observed={r["gesture_label"] for r in target}
    configured=experiment.get("target_gesture_labels")
    labels=sorted(observed if configured is None else configured)
    if not labels or len(labels)!=len(set(labels)) or any(not isinstance(label,str) or not label for label in labels):
        raise ValueError("target_gesture_labels must be a nonempty unique list, or null")
    missing=set(labels)-{r["gesture_label"] for r in records}
    if missing:
        raise ValueError("Configured labels absent from annotations: " + ", ".join(sorted(missing)))
    eligible=[r for r in records if r["gesture_label"] in labels]
    target=[r for r in target if r["gesture_label"] in labels]
    reasons={r["record_id"]:"gesture_outside_target_vocabulary" for r in records if r["gesture_label"] not in labels}
    gesture_labels = labels
    if task == "gesture_object":
        labels = sorted({label_of(r) for r in target})
        reasons.update({r["record_id"]: "gesture_object_outside_target_vocabulary"
                        for r in eligible if label_of(r) not in labels})
        eligible = [r for r in eligible if label_of(r) in labels]
    settings=deepcopy(protocol.get("validation") or {})
    protocol_name=protocol.get("name", "protocol1" if protocol.get("type") in SUBJECT_PROTOCOL_TYPES else "protocol2")
    metadata=dict(experiment=name,train_sessions=list(train_sessions),test_sessions=["step4"],
                  task=task, target_gesture_labels=gesture_labels, target_class_labels=labels,
                  seed=seed,split_before_windowing=True,
                  macro_grouping="subject/session/assembly_repetition_prefix",
                  source_only_labels=sorted({r["gesture_label"] for r in records}-set(gesture_labels)),
                  source_only_class_labels=sorted({label_of(r) for r in records}-set(labels)))
    if protocol.get("type") == "grouped_percentages":
        percentages, split_seed, source_validation = percentage_settings
        macro_parts, macro_meta = _percentage_macro_partitions(target, percentages, split_seed)
        source = [r for r in eligible if r["session_id"] in train_sessions and r["session_id"] != "step4"]
        if "step4" in train_sessions:
            assigned = dict(train=macro_parts["train"] + source,
                            validation=macro_parts["validation"], test=macro_parts["test"])
            val_sessions, valmeta = ["step4"], {"strategy": "macro_percentage_partition"}
        else:
            train, validation, valmeta = _validation_pool(source, labels, source_validation, split_seed)
            assigned = dict(train=train, validation=validation, test=macro_parts["test"])
            val_sessions = list(train_sessions)
            for part in ("train", "validation"):
                reasons.update({r["record_id"]: "step4_not_used_for_source_only_training" for r in macro_parts[part]})
        meta = dict(metadata, protocol=protocol_name, protocol_type="grouped_percentages",
                    macro_percentage_split=macro_meta, validation=valmeta, validation_sessions=val_sessions,
                    participant_scope="assemblies pooled across participants; not an unseen-participant protocol",
                    step4_train_validation_used="step4" in train_sessions)
        fold_reasons = _apply_source_sampling(assigned, meta, reasons, experiment, seed)
        return [_finish(protocol_name, records, assigned, meta, labels, fold_reasons)]
    if protocol.get("type") == "held_out_objects":
        from macro_tools.object_splits import build_object_split
        return [build_object_split(records, eligible, target, labels, experiment, protocol,
                                   metadata, reasons, seed)]
    if protocol.get("type") in SUBJECT_PROTOCOL_TYPES:
        subjects=sorted({r["subject_id"] for r in target})
        folds=[]
        for selection in participant_partitions(subjects, protocol):
            val_sessions=["step4"] if "step4" in train_sessions else list(train_sessions)
            assigned=dict(train=[r for r in eligible if r["session_id"] in train_sessions and r["subject_id"] in selection["train"]],
                          validation=[r for r in eligible if r["subject_id"] in selection["validation"] and r["session_id"] in val_sessions],
                          test=[r for r in target if r["subject_id"] in selection["test"]])
            for part in PARTS:
                absent=set(selection[part])-{r["subject_id"] for r in assigned[part]}
                if absent:
                    raise ValueError("Selected {} participants have no eligible annotations in the required context: {}".format(part, ", ".join(sorted(absent))))
            meta=dict(metadata, protocol=protocol_name, protocol_type=protocol["type"],
                      subject_selection=selection["selection"], subject_counts={p:len(selection[p]) for p in PARTS},
                      requested_subject_counts=selection["requested_counts"],
                      train_subject_ids=selection["train"], validation_subject_ids=selection["validation"],
                      test_subject_ids=selection["test"], excluded_subject_ids=selection["excluded"],
                      test_subject_id=selection["test"][0] if len(selection["test"])==1 else None,
                      validation_subject_id=selection["validation"][0] if len(selection["validation"])==1 else None,
                      validation_sessions=val_sessions)
            fold_reasons=dict(reasons, **{r["record_id"]:"participant_not_selected"
                                         for r in eligible if r["subject_id"] in selection["excluded"]})
            fold_reasons = _apply_source_sampling(assigned, meta, fold_reasons, experiment, seed)
            folds.append(_finish(protocol_name+"_"+"-".join(selection["test"]),records,assigned,meta,labels,fold_reasons))
        return folds
    if protocol.get("type") in ("held_out_repetitions","held_out_macro_repetitions"):
        train_reps=_ids(protocol.get("train_repetitions",[1,2]),"train_repetitions")
        test_reps=_ids(protocol.get("test_repetitions",[3]),"test_repetitions")
        if train_reps & test_reps:
            raise ValueError("Macro train/test repetitions must be disjoint")
        test=[r for r in target if r["repetition"] in test_reps]
        source=[r for r in eligible if r["session_id"] in train_sessions and r["session_id"]!="step4"]
        if "step4" in train_sessions:
            pool=[r for r in target if r["repetition"] in train_reps]
            pool_labels=sorted({label_of(r) for r in pool})
            train,validation,valmeta=_validation_pool(pool,pool_labels,settings,seed)
            train+=source
            val_sessions=["step4"]
        else:
            train,validation,valmeta=_validation_pool(source,labels,settings,seed)
            val_sessions=list(train_sessions)
        assigned=dict(train=train,validation=validation,test=test)
        meta=dict(metadata,protocol=protocol_name,protocol_type="held_out_macro_repetitions",
                  train_repetitions=sorted(train_reps),test_repetitions=sorted(test_reps),
                  validation=valmeta,validation_sessions=val_sessions,
                  participant_scope="participants may appear in both training and test")
        fold_reasons = _apply_source_sampling(assigned, meta, reasons, experiment, seed)
        return [_finish(protocol_name,records,assigned,meta,labels,fold_reasons)]
    raise ValueError("Unknown protocol type: " + str(protocol.get("type")))
