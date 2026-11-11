"""Joint targets, whole-assembly isolation, adapters and final reports on invented data."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from macro_tools import cli, training
from macro_tools.adapters import data
from macro_tools.metrics import classification_metrics, aggregate_fold_metrics
from macro_tools.paths import PROJECT_ROOT, experiment_run_name
from macro_tools.splits import build_splits, trial_key
from macro_tools.tasks import joint_label, label_of, split_joint_label
from test_object_splits import records, protocol as object_protocol
from test_training import InventedClips, make_config


def repetition_protocol():
    return dict(name="protocol2", type="held_out_repetitions", train_repetitions=[1, 2],
                test_repetitions=[3], validation=dict(strategy="grouped_trials", fraction=.2))


class JointTaskTests(unittest.TestCase):
    def test_all_experiments_protocols_cover_30_pairs_and_keep_trials_isolated(self):
        original = records()
        before = copy.deepcopy(original)
        p2 = repetition_protocol()
        protocols = [p2, dict(p2, grouped_percentages=True,
                     split_percentages=dict(train=70, validation=15, test=15), split_seed=42),
                     dict(name="protocol1", type="subject_holdout",
                          subject_counts=dict(train=None, validation=1, test=1))]
        for protocol in protocols:
            reference = None
            for name in ("step4_to_step4", "steps1234_to_step4", "steps123_to_step4"):
                experiment = dict(name=name)
                if name != "step4_to_step4":
                    experiment["steps123_train_percent"] = 20
                folds = build_splits(original, experiment, protocol, task="gesture_object")
                test_ids = [{r["record_id"] for r in f["test"]} for f in folds]
                if reference is None:
                    reference = test_ids
                self.assertEqual(test_ids, reference)
                for fold in folds:
                    owners = {}
                    for part in ("train", "validation", "test"):
                        self.assertEqual(len({label_of(r) for r in fold[part]}), 30)
                        for row in fold[part]:
                            self.assertEqual(split_joint_label(label_of(row)), (row["gesture_label"], row["object_id"]))
                            self.assertEqual(owners.setdefault(trial_key(row), part), part)
                            self.assertNotEqual(row["object_id"], "6")
                    self.assertEqual(len(fold["metadata"]["target_gesture_labels"]), 7)
                    self.assertEqual(len(fold["metadata"]["target_class_labels"]), 30)
                    if name != "step4_to_step4":
                        self.assertTrue(any(r["exclusion_reason"] == "gesture_object_outside_target_vocabulary"
                                            and r["object_id"] == "6" for r in fold["excluded"]))
        self.assertEqual(original, before)
        gesture = build_splits(original, "step4_to_step4", p2)[0]
        joint = build_splits(original, "step4_to_step4", p2, task="gesture_object")[0]
        for part in ("train", "validation", "test"):
            self.assertEqual({r["record_id"] for r in gesture[part]}, {r["record_id"] for r in joint[part]})

    def test_missing_source_class_is_not_silently_removed_and_protocol3_rejected(self):
        rows = [r for r in records() if not (r["session_id"] != "step4" and r["object_id"] == "5")]
        with self.assertRaisesRegex(ValueError, "class"):
            build_splits(rows, "steps123_to_step4", repetition_protocol(), task="gesture_object")
        with self.assertRaisesRegex(ValueError, "Protocol3"):
            build_splits(records(), "step4_to_step4", object_protocol(), task="gesture_object")
        with patch("macro_tools.annotations.load_annotations") as load, \
             patch.object(cli, "create_run_directory") as create, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main("imu", ["--task", "gesture_object", "--protocol", "protocol3", "--audit-only"]), 2)
        load.assert_not_called()
        create.assert_not_called()

    def test_cli_task_config_override_and_names_for_all_modalities(self):
        base = PROJECT_ROOT / "runs"
        base.mkdir(exist_ok=True)
        for modality in ("imu", "semg", "rgb", "depth"):
            for configured, flags, expected in (("gesture", ["--task", "gesture_object"], "gesture_object"),
                                                ("gesture_object", [], "gesture_object"),
                                                ("gesture_object", ["--task", "gesture"], "gesture")):
                configs = [dict(modality=modality, model="invented", task=configured),
                           dict(name="step4_to_step4"), repetition_protocol(),
                           dict(dataset_root="../dataset", read_only=True)]
                with tempfile.TemporaryDirectory(prefix="joint_cli_", dir=base) as folder:
                    with patch.object(cli, "read_config", side_effect=configs), \
                         patch.object(cli, "create_run_directory", return_value=Path(folder)), \
                         patch.object(cli, "source_versions", return_value={}), \
                         patch("macro_tools.annotations.load_annotations", return_value=records()), \
                         contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(cli.main(modality, flags + ["--audit-only"]), 0)
                    saved = json.loads((Path(folder) / "config.json").read_text())
                    self.assertEqual(saved["config"]["task"], expected)
                    self.assertEqual(len(saved["preparation"]["label_mapping"]), 30 if expected == "gesture_object" else 7)
                    self.assertEqual({p.name for p in Path(folder).iterdir()}, {"config.json"})
            self.assertTrue(experiment_run_name(modality, dict(task="gesture_object"), repetition_protocol())
                            .startswith(modality + "_gesture_object_"))

    def test_signal_targets_preserve_raw_gestures_and_training_only_scaling(self):
        fold = build_splits(records(), "step4_to_step4", repetition_protocol(), task="gesture_object")[0]
        assigned = data._assigned([fold])
        labels = {name: i for i, name in enumerate(sorted({label_of(r) for r in assigned}))}
        for modality, shape, axes in (("imu", (1, 18, 16), (0, 1, 3)), ("semg", (8, 16, 16), (0, 2))):
            train_ids = {r["record_id"] for r in fold["train"]}
            windows = {r["record_id"]: [(np.full(shape, 1 if r["record_id"] in train_ids else 101,
                                                        dtype=np.float32), data._provenance(r))] for r in assigned}
            with patch.object(data, "_signal_examples", return_value=(windows, shape, axes)):
                prepared = data._signal_folds(assigned, [fold], {}, dict(modality=modality), labels, [], [])[0]
            for part in ("train", "validation", "test"):
                np.testing.assert_array_equal(prepared["splits"][part]["y"], [labels[label_of(r)] for r in fold[part]])
                for row in prepared["splits"][part]["metadata"]:
                    self.assertEqual(split_joint_label(row["target_label"])[0], row["gesture_label"])
            np.testing.assert_array_equal(prepared["splits"]["train"]["x"], 0)
            np.testing.assert_array_equal(prepared["splits"]["test"]["x"], 100)

    def test_exact_pair_metrics_differ_from_components(self):
        labels = {joint_label("grasp", "1"): 0, joint_label("grasp", "2"): 1, joint_label("insert", "2"): 2}
        metric = classification_metrics([0, 1, 2], [1, 1, 1], labels)
        self.assertAlmostEqual(metric["accuracy"], 1/3)
        self.assertAlmostEqual(metric["component_metrics"]["gesture"]["accuracy"], 2/3)
        self.assertAlmostEqual(metric["component_metrics"]["object"]["accuracy"], 2/3)
        self.assertEqual(aggregate_fold_metrics([metric, metric])["component_metrics"]["object"]["accuracy"]["std"], 0)
        self.assertEqual(split_joint_label(joint_label("x::object=y", "a/b")), ("x::object=y", "a/b"))

    def test_30_class_training_checkpoint_predictions_and_plots(self):
        labels = {name: i for i, name in enumerate(sorted({joint_label(r["gesture_label"], r["object_id"])
                                                        for r in records() if r["session_id"] == "step4"}))}
        class TinyJoint(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(1, 30)
            def forward(self, x):
                return self.classifier(x.mean(dim=(1, 2, 3, 4, 5)).unsqueeze(1))
        config = make_config()
        config.update(task="gesture_object", synthetic=True)
        config["training"].update(epochs=1, batch_size=30)
        fold = dict(name="synthetic_joint", splits={p: [dict(label=i, partition=p, sample_id=f"{p}_{i}")
                                                       for i in range(30)] for p in ("train", "validation", "test")})
        (PROJECT_ROOT / "runs" / "rgb").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="joint_training_", dir=PROJECT_ROOT / "runs" / "rgb") as folder:
            stdout = io.StringIO()
            with patch.object(training, "_make_dataset", side_effect=InventedClips), \
                 patch.object(training, "_make_model", side_effect=lambda *_: TinyJoint()), \
                 patch.object(training, "_model_info", return_value={"synthetic": True}), \
                 contextlib.redirect_stdout(stdout):
                result = training.train_fold(fold, labels, config, folder, "unused")
            self.assertEqual(len(result["test_metrics"]["confusion_matrix"]), 30)
            self.assertIn("exact gesture-object pair", stdout.getvalue())
            payload = torch.load(Path(folder) / "best.pt", weights_only=True, map_location="cpu")
            self.assertEqual(payload["label_mapping"], labels)
            self.assertEqual(payload["config"]["task"], "gesture_object")
            with (Path(folder) / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))
            for row in saved:
                self.assertEqual(split_joint_label(row["target_label"]), (row["target_gesture"], row["target_object"]))
                self.assertEqual(split_joint_label(row["predicted_label"]), (row["predicted_gesture"], row["predicted_object"]))
            self.assertFalse((Path(folder) / "plot_error.txt").exists())
            self.assertTrue((Path(folder) / "plots" / "confusion_matrix.png").exists())
            self.assertIn("gesture_object", (Path(folder) / "report.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
