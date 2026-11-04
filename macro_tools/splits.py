"""Create source-context experiments with participant or whole-trial holdouts.

The prediction target remains a micro-gesture label. Every test record is from
step4. The three experiment names control training sessions independently of the
participant/repetition protocol. All assignments happen before windowing.

Protocol1 excludes its test and validation participants from every training
session. Protocol2 uses macro repetition prefixes (default train 1-2, test 3)
and reserves complete subject/whole-assembly groups for validation.
Experiment 2 adds isolated-context data to experiment 1's training partition;
experiment 3 uses only isolated-context training/validation, retaining the same
step4 test population. By default, class vocabulary is the observed step4
vocabulary; source-only labels are explicitly listed as exclusions, never renamed.
Overlapping annotated time intervals are recorded as warnings under the user's
accepted policy. Reused record IDs and trial groups across partitions are errors.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import random

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
    classes = {key: {record["gesture_label"] for record in groups[key]} for key in keys}
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
        missing=set(labels)-{record["gesture_label"] for record in fold[part]}
        if missing and part=="train":
            raise ValueError("Training is missing target classes: " + ", ".join(sorted(missing)))
        if missing and fold[part] and part in ("validation","test"):
            fold["issues"].append(dict(code="missing_"+part+"_classes",severity="warning",
                message="The partition has no examples for some fixed target classes",missing_classes=sorted(missing)))
    if not fold["validation"]:
        fold["issues"].append(dict(code="no_validation_split",severity="warning",message="Last epoch will be used; no validation tuning"))
    fold["issues"].extend(_overlap_warnings(fold))
    fold["summary"]={part:dict(segments=len(fold[part]), trials=len({trial_key(r) for r in fold[part]}),
        subjects=sorted({r["subject_id"] for r in fold[part]}), sessions=sorted({r["session_id"] for r in fold[part]}),
        gesture_counts=dict(sorted(Counter(r["gesture_label"] for r in fold[part]).items()))) for part in PARTS+("excluded",)}
    fold["valid"]=True
    return fold


def build_splits(records, experiment, protocol, seed=42):
    """Return macro-context folds; input records/configuration remain unchanged."""
    records=deepcopy(list(records))
    if not records:
        raise ValueError("No annotation records")
    ids=[r["record_id"] for r in records]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate source record IDs")
    experiment={"name":experiment} if isinstance(experiment,str) else deepcopy(experiment)
    name=experiment.get("name")
    if name not in EXPERIMENTS:
        raise ValueError("Unknown macro experiment: " + str(name))
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
    settings=deepcopy(protocol.get("validation") or {})
    validation_none=settings.get("strategy")=="none"
    protocol_name=protocol.get("name", "protocol1" if protocol.get("type")=="leave_one_subject_out" else "protocol2")
    metadata=dict(experiment=name,train_sessions=list(train_sessions),test_sessions=["step4"],
                  target_gesture_labels=labels,seed=seed,split_before_windowing=True,
                  macro_grouping="subject/session/assembly_repetition_prefix",
                  source_only_labels=sorted({r["gesture_label"] for r in records}-set(labels)))
    if protocol.get("type")=="leave_one_subject_out":
        if settings.get("strategy","auto") not in ("auto","next_subject","none"):
            raise ValueError("Participant validation must use next_subject or none")
        subjects=sorted({r["subject_id"] for r in target})
        tests=subjects if protocol.get("test_subject_id") is None else [protocol["test_subject_id"]]
        if len(subjects)<(2 if validation_none else 3):
            raise ValueError("Participant holdout needs at least two subjects, or three with validation")
        folds=[]
        for test in tests:
            if test not in subjects:
                raise ValueError("Unknown step4 test participant: " + str(test))
            validation=None if validation_none else settings.get("subject_id", protocol.get("validation_subject_id")) or subjects[(subjects.index(test)+1)%len(subjects)]
            if validation is not None and (validation not in subjects or validation==test):
                raise ValueError("Validation participant must be available and differ from test participant")
            val_sessions=["step4"] if "step4" in train_sessions else list(train_sessions)
            assigned=dict(train=[r for r in eligible if r["session_id"] in train_sessions and r["subject_id"] not in (test,validation)],
                          validation=[r for r in eligible if validation is not None and r["subject_id"]==validation and r["session_id"] in val_sessions],
                          test=[r for r in target if r["subject_id"]==test])
            meta=dict(metadata, protocol=protocol_name,protocol_type="leave_one_subject_out",test_subject_id=test,
                      validation_subject_id=validation,validation_sessions=val_sessions)
            folds.append(_finish(protocol_name+"_"+test,records,assigned,meta,labels,reasons))
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
            pool_labels=sorted({r["gesture_label"] for r in pool})
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
        return [_finish(protocol_name,records,assigned,meta,labels,reasons)]
    raise ValueError("Unknown protocol type: " + str(protocol.get("type")))
