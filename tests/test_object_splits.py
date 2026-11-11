"""Protocol3 tests use invented annotations, never source recordings."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from macro_tools import cli
from macro_tools.paths import PROJECT_ROOT, experiment_run_name
from macro_tools.splits import build_splits, trial_key

LABELS = ("approach", "grasp", "lift", "move", "release", "screw", "insert")
PARTS = ("train", "validation", "test")


def protocol():
    return dict(name="protocol3", type="held_out_objects", split_before_windowing=True,
                test_object_ids={g: ["2" if g == "screw" else "5"] for g in LABELS},
                assembly_split=dict(name="assemblies", type="held_out_repetitions",
                    split_before_windowing=True, grouped_percentages=False,
                    train_repetitions=[1, 2], test_repetitions=[3],
                    validation=dict(strategy="grouped_trials", fraction=.2),
                    split_percentages=dict(train=70, validation=15, test=15), split_seed=42),
                source_validation=dict(strategy="grouped_trials", fraction=.2))


def records():
    rows = []
    for subject in ("a", "b", "c", "d"):
        for rep in (1, 2, 3):
            for obj, count in (("1", 3), ("2", 3), ("3", 3), ("4", 2), ("5", 1)):
                gestures = LABELS[:5] + (("screw",) if obj in ("1", "2") else ("insert",))
                for sub in range(1, count + 1):
                    for gesture in gestures:
                        rows.append(dict(record_id=f"{subject}-4-{rep}-{obj}-{sub}-{gesture}",
                                         subject_id=subject, session_id="step4", object_id=obj,
                                         repetition=rep, macro_substep=sub, gesture_label=gesture, intervals={}))
        for gesture in LABELS:
            session = "step2" if gesture == "screw" else "step3" if gesture == "insert" else "step1"
            objects = ("1", "2") if gesture == "screw" else ("3", "4", "5") if gesture == "insert" else ("1", "2", "3", "4", "5", "6")
            for obj in objects:
                for rep in (1, 2, 3, 4, 5):
                    rows.append(dict(record_id=f"{subject}-{session}-{gesture}-{obj}-{rep}",
                                     subject_id=subject, session_id=session, object_id=obj, repetition=rep,
                                     gesture_label=gesture, intervals={}))
    return rows


class ObjectSplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = records()
        self.config = protocol()

    def split(self, experiment="step4_to_step4", config=None, rows=None):
        return build_splits(self.rows if rows is None else rows, experiment, config or self.config)[0]

    def assert_isolated(self, fold, config):
        owners, ids = {}, set()
        for part in PARTS:
            self.assertEqual({r["gesture_label"] for r in fold[part]}, set(LABELS))
            for row in fold[part]:
                self.assertNotIn(row["record_id"], ids)
                ids.add(row["record_id"])
                self.assertEqual(owners.setdefault(trial_key(row), part), part)
                self.assertEqual(row["object_id"] in config["test_object_ids"][row["gesture_label"]], part == "test")
                if part == "test":
                    self.assertEqual(row["session_id"], "step4")
        excluded = {r["record_id"] for r in fold["excluded"]}
        self.assertFalse(ids & excluded)
        self.assertEqual(ids | excluded, {r["record_id"] for r in self.rows})

    def test_defaults_count_examples_and_do_not_mutate_inputs(self):
        before = deepcopy((self.rows, self.config))
        fold = self.split()
        self.assert_isolated(fold, self.config)
        self.assertEqual([len(fold[p]) for p in PARTS], [378, 126, 36])
        self.assertEqual(fold["metadata"]["assembly_counts_before_object_filter"], dict(train=6, validation=2, test=4))
        self.assertEqual(fold["summary"]["test"]["gesture_counts"]["screw"], 12)
        self.assertEqual(fold["summary"]["test"]["gesture_counts"]["grasp"], 4)
        self.assertEqual((self.rows, self.config), before)
        self.assertEqual(len(build_splits(self.rows, "step4_to_step4", self.config)), 1)

    def test_same_test_population_for_all_experiments_and_source_sampling(self):
        first = self.split()
        for name in ("steps1234_to_step4", "steps123_to_step4"):
            for percent in (25, 100):
                fold = self.split(dict(name=name, steps123_train_percent=percent))
                self.assert_isolated(fold, self.config)
                self.assertEqual(fold["test"], first["test"])
                self.assertIn("held_out_object_in_source", fold["metadata"]["excluded_reason_counts"])
                if name == "steps123_to_step4":
                    self.assertFalse(any(r["session_id"] == "step4" for p in ("train", "validation") for r in fold[p]))
                else:
                    self.assertEqual(fold["validation"], first["validation"])
                # Per-gesture holdout permits object2 for general gestures.
                self.assertTrue(any(r["object_id"] == "2" and r["gesture_label"] == "grasp" for r in fold["train"]))

    def test_custom_objects_change_test_set_and_apply_to_source_data(self):
        config = deepcopy(self.config)
        config["test_object_ids"] = {g: ["1" if g == "screw" else "3" if g == "insert" else "4"] for g in LABELS}
        fold = self.split("steps1234_to_step4", config)
        self.assert_isolated(fold, config)
        self.assertNotEqual(fold["test"], self.split()["test"])
        self.assertTrue(all(r["object_id"] == "4" for r in fold["test"] if r["gesture_label"] == "grasp"))

    def test_percentage_assembly_split_is_reproducible_and_ignores_repetition_ids(self):
        config = deepcopy(self.config)
        config["assembly_split"]["grouped_percentages"] = True
        config["assembly_split"]["train_repetitions"] = [999]
        config["assembly_split"]["test_repetitions"] = [999]
        first = self.split(config=config)
        self.assert_isolated(first, config)
        again = self.split(config=config, rows=list(reversed(self.rows)))
        for part in PARTS:
            self.assertEqual({r["record_id"] for r in first[part]}, {r["record_id"] for r in again[part]})
        self.assertEqual(first["metadata"]["assembly_counts_before_object_filter"], dict(train=8, validation=2, test=2))
        for name in ("steps1234_to_step4", "steps123_to_step4"):
            self.assertEqual(first["test"], self.split(name, config)["test"])

    def test_invalid_choices_and_removed_class_coverage_fail(self):
        modifications = [
            {"screw": ["9"]}, {"screw": ["1", "2"]}, {"screw": [2]}, {"screw": "2"},
            {"screw": []}, {"screw": ["2", "2"]}, {"unknown": ["5"]},
        ]
        for changed in modifications:
            config = deepcopy(self.config)
            config["test_object_ids"].update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.split(config=config)
        config = deepcopy(self.config)
        del config["test_object_ids"]["insert"]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.split(config=config)
        config = deepcopy(self.config)
        config["assembly_split"]["type"] = "held_out_objects"
        with self.assertRaisesRegex(ValueError, "assembly_split"):
            self.split(config=config)
        filtered = [r for r in self.rows if not (
            r["session_id"] == "step4" and r["repetition"] == 3 and r["gesture_label"] == "insert")]
        with self.assertRaisesRegex(ValueError, "test lacks gestures"):
            self.split(rows=filtered)

    def test_disabled_validation_and_excluded_repetitions(self):
        config = deepcopy(self.config)
        config["assembly_split"]["validation"] = {"strategy": "none"}
        config["source_validation"] = {"strategy": "none"}
        config["assembly_split"]["train_repetitions"] = [1]
        for name in ("step4_to_step4", "steps123_to_step4"):
            fold = self.split(name, config)
            self.assertFalse(fold["validation"])
            self.assertTrue(all(r["repetition"] == 3 for r in fold["test"]))
            self.assertIn("outside_assembly_split", fold["metadata"]["excluded_reason_counts"])

    def test_run_names_distinguish_per_gesture_rules_and_assembly_settings(self):
        name = experiment_run_name("imu", {}, self.config)
        self.assertIn("objects2-5-", name)
        first, second = deepcopy(self.config), deepcopy(self.config)
        first["test_object_ids"]["grasp"] = ["4"]
        second["test_object_ids"]["lift"] = ["4"]
        self.assertNotEqual(experiment_run_name("imu", {}, first), experiment_run_name("imu", {}, second))
        first = deepcopy(self.config)
        first["assembly_split"]["grouped_percentages"] = True
        self.assertNotEqual(name, experiment_run_name("imu", {}, first))

    def test_cli_short_name_and_compact_saved_settings_for_all_modalities(self):
        base = PROJECT_ROOT / "runs" / "imu"
        base.mkdir(parents=True, exist_ok=True)
        for modality in ("imu", "semg", "rgb", "depth"):
            configs = [dict(modality=modality, model="invented", training=dict(seed=42)),
                       {"name": "step4_to_step4"}, self.config,
                       {"dataset_root": "../dataset", "read_only": True}]
            with tempfile.TemporaryDirectory(prefix="object_cli_", dir=base) as folder:
                output = Path(folder)
                stdout = io.StringIO()
                with patch.object(cli, "read_config", side_effect=deepcopy(configs)) as read, \
                     patch.object(cli, "create_run_directory", return_value=output), \
                     patch.object(cli, "source_versions", return_value={}), \
                     patch("macro_tools.annotations.load_annotations", return_value=self.rows), \
                     contextlib.redirect_stdout(stdout):
                    self.assertEqual(cli.main(modality, ["--protocol", "protocol3", "--audit-only"]), 0)
                self.assertEqual(read.call_args_list[2].args[0], "configs/protocols/protocol3.json")
                saved = json.loads((output / "config.json").read_text())
                self.assertEqual(saved["protocol"]["test_object_ids"], self.config["test_object_ids"])
                self.assertEqual(len(saved["preparation"]["folds"]), 1)
                self.assertEqual({f.name for f in output.iterdir()}, {"config.json"})
                self.assertIn("screw=2", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
