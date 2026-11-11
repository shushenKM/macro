"""Classification targets for macro-context recognition, independent of modality.

gesture keeps the seven step4 gestures. gesture_object uses only observed step4
pairs (30 in the current dataset), excluding source-only pairs before splitting.
Raw gesture/object fields stay intact for preprocessing and assembly grouping.
Joint classification is closed-set: use protocol1 or protocol2, including its
percentage mode. Protocol3 deliberately holds out classes and is gesture-only.
All labels are derived in memory; annotations are never rewritten.
"""
from urllib.parse import quote, unquote

TASKS = ("gesture", "gesture_object")
SEPARATOR = "::object="


def validate_task(task="gesture", protocol=None):
    if task not in TASKS:
        raise ValueError("task must be gesture or gesture_object")
    if task == "gesture_object" and (protocol or {}).get("type") == "held_out_objects":
        raise ValueError("Protocol3 supports task=gesture only: held-out objects create unseen "
                         "gesture-object classes. Use protocol1 or protocol2 for gesture_object.")
    return task


def joint_label(gesture, object_id):
    if any(not isinstance(value, str) or not value.strip() for value in (gesture, object_id)):
        raise ValueError("A gesture-object label needs a nonempty gesture and object ID")
    return quote(gesture.strip(), safe="") + SEPARATOR + quote(object_id.strip(), safe="")


def split_joint_label(label):
    gesture, separator, obj = label.partition(SEPARATOR)
    if not separator or not gesture or not obj:
        raise ValueError("Invalid gesture-object label: " + str(label))
    return unquote(gesture), unquote(obj)


def label_of(record):
    return record.get("target_label", record["gesture_label"])


def with_task_labels(records, task="gesture", protocol=None):
    validate_task(task, protocol)
    return [dict(row, target_label=joint_label(row["gesture_label"], row["object_id"])
                 if task == "gesture_object" else row["gesture_label"]) for row in records]


def prediction_labels(target, prediction):
    if SEPARATOR in target and SEPARATOR in prediction:
        tg, to = split_joint_label(target)
        pg, po = split_joint_label(prediction)
        return dict(target_label=target, predicted_label=prediction, target_gesture=tg,
                    predicted_gesture=pg, target_object=to, predicted_object=po)
    return dict(target_gesture=target, predicted_gesture=prediction)
