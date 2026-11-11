"""Percentage allocation keeps assemblies intact and transfer tests comparable."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from macro_tools import cli
from macro_tools.paths import PROJECT_ROOT, experiment_run_name
from macro_tools.splits import build_splits, trial_key
from test_splits import records


class PercentageSplitTests(unittest.TestCase):
    def test_unified_switch_matches_existing_modes_and_keeps_transfer_behavior(self):
        from macro_tools.paths import read_config
        from macro_tools.protocol_config import normalize_protocol, resolve_protocol_reference
        unified = read_config("configs/protocols/protocol2.json")
        self.assertIs(unified["grouped_percentages"], False)
        original = copy.deepcopy(unified)
        old_repetitions = {k: v for k, v in unified.items()
                           if k not in ("grouped_percentages", "split_percentages", "split_seed", "source_validation")}
        for experiment in ("step4_to_step4", "steps1234_to_step4", "steps123_to_step4"):
            self.assertEqual(self.split(experiment, unified), self.split(experiment, old_repetitions))
            switched = dict(unified, grouped_percentages=True)
            actual = self.split(experiment, switched)
            old = self.split(experiment)
            for part in ("train", "validation", "test"):
                self.assertEqual(actual[part], old[part])
            self.assertEqual(actual["metadata"]["macro_percentage_split"], old["metadata"]["macro_percentage_split"])
            self.assertEqual(actual["name"], "protocol2")
            self.assertIn("split70-15-15", experiment_run_name("imu", {}, switched))
        self.assertEqual(unified, original)
        path, force = resolve_protocol_reference("protocol2_percentages", "configs/protocols")
        self.assertEqual(path, "configs/protocols/protocol2.json")
        self.assertEqual(normalize_protocol(unified, force_percentages=force)["type"], "grouped_percentages")
        for flag in (None, "true", 1):
            with self.assertRaisesRegex(ValueError, "grouped_percentages"):
                self.split(protocol=dict(unified, grouped_percentages=flag))

    def setUp(self):
        self.records = records()
        # Fixed fixture: users can edit their experiment percentages freely.
        self.protocol = dict(name="protocol2_percentages", type="grouped_percentages",
                             split_before_windowing=True, split_seed=42,
                             split_percentages=dict(train=70, validation=15, test=15),
                             source_validation=dict(strategy="grouped_trials", fraction=.2))

    def split(self, experiment="step4_to_step4", protocol=None, records_=None):
        return build_splits(self.records if records_ is None else records_, experiment,
                            self.protocol if protocol is None else protocol)[0]

    def test_group_counts_partition_all_rows_and_do_not_fix_repetition_three(self):
        before = copy.deepcopy(self.records)
        fold = self.split()
        meta = fold["metadata"]["macro_percentage_split"]
        self.assertEqual(meta["assembly_counts"], {"train": 8, "validation": 2, "test": 2})
        assigned = set()
        owners = {}
        for part in ("train", "validation", "test"):
            for row in fold[part]:
                self.assertNotIn(row["record_id"], assigned)
                assigned.add(row["record_id"])
                key = trial_key(row)
                self.assertEqual(owners.setdefault(key, part), part)
        self.assertEqual(assigned, {r["record_id"] for r in self.records if r["session_id"] == "step4"})
        self.assertTrue(any(r["repetition"] != 3 for r in fold["test"]))
        self.assertTrue(any(r["repetition"] == 3 for r in fold["train"]))
        self.assertEqual(before, self.records)

    def test_unequal_object_occurrences_stay_in_the_same_complete_assembly(self):
        # Make one object's sub-actions unequal without changing the assembly key.
        rows = [r for r in self.records if not (r["subject_id"] == "a" and r["session_id"] == "step4" and r["object_id"] == "2" and r["gesture_label"] == "insert")]
        fold = self.split(records_=rows)
        meta = fold["metadata"]["macro_percentage_split"]
        self.assertEqual(sum(meta["assembly_counts"].values()), 12)
        total = sum(len(fold[p]) for p in ("train", "validation", "test"))
        for part in ("train", "validation", "test"):
            self.assertAlmostEqual(meta["actual_annotation_percentages"][part], 100 * len(fold[part]) / total)
            keys = {trial_key(r) for r in fold[part]}
            expected = {r["record_id"] for r in rows if r["session_id"] == "step4" and trial_key(r) in keys}
            self.assertEqual(expected, {r["record_id"] for r in fold[part]})

    def test_three_experiments_and_source_percentages_share_test_population(self):
        baseline = self.split()
        for name in ("steps1234_to_step4", "steps123_to_step4"):
            for percent in (25, 50, 100):
                fold = self.split({"name": name, "steps123_train_percent": percent})
                self.assertEqual(fold["test"], baseline["test"])
                self.assertEqual(fold["metadata"]["macro_percentage_split"], baseline["metadata"]["macro_percentage_split"])
                if name == "steps1234_to_step4":
                    self.assertEqual(fold["validation"], baseline["validation"])
                    self.assertEqual([r for r in fold["train"] if r["session_id"] == "step4"], baseline["train"])
                else:
                    self.assertTrue(all(r["session_id"] != "step4" for p in ("train", "validation") for r in fold[p]))
                    self.assertTrue(any(r["exclusion_reason"] == "step4_not_used_for_source_only_training" for r in fold["excluded"]))

    def test_reproducibility_row_order_and_split_seed(self):
        original = self.split()
        self.assertEqual(original, self.split())
        reversed_rows = self.split(records_=list(reversed(self.records)))
        self.assertEqual(original["metadata"]["macro_percentage_split"], reversed_rows["metadata"]["macro_percentage_split"])
        other = self.split(protocol=dict(self.protocol, split_seed=7))
        self.assertNotEqual({r["record_id"] for r in original["test"]}, {r["record_id"] for r in other["test"]})

    def test_changed_percentages_zero_validation_and_small_population(self):
        changed = dict(self.protocol, split_percentages={"train": 50, "validation": 25, "test": 25})
        self.assertEqual(self.split(protocol=changed)["metadata"]["macro_percentage_split"]["assembly_counts"],
                         {"train": 6, "validation": 3, "test": 3})
        no_val = dict(self.protocol, split_percentages={"train": 80, "validation": 0, "test": 20})
        self.assertFalse(self.split(protocol=no_val)["validation"])
        one_subject = [r for r in self.records if r["subject_id"] == "a"]
        self.assertEqual(self.split(records_=one_subject)["metadata"]["macro_percentage_split"]["assembly_counts"],
                         {"train": 1, "validation": 1, "test": 1})
        too_few = [r for r in one_subject if r.get("repetition") != 3]
        with self.assertRaisesRegex(ValueError, "complete assemblies"):
            self.split(records_=too_few)

    def test_invalid_percentages_and_conflicting_settings_rejected(self):
        invalid = [None, {}, {"train": 70, "validation": 15},
                   {"train": 70, "validation": 15, "test": 20},
                   {"train": 0, "validation": 50, "test": 50},
                   {"train": 50, "validation": 50, "test": 0},
                   {"train": 80, "validation": -5, "test": 25}]
        invalid += [{"train": v, "validation": 15, "test": 15} for v in (True, "70", float("nan"), float("inf"))]
        for value in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "split_percentages"):
                self.split(protocol=dict(self.protocol, split_percentages=value))
        for field in ("train_repetitions", "test_repetitions", "validation"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "grouped_percentages"):
                self.split(protocol=dict(self.protocol, **{field: [1]}))
        for seed in (True, -1, 1.5, None):
            with self.assertRaisesRegex(ValueError, "split_seed"):
                self.split(protocol=dict(self.protocol, split_seed=seed))

    def test_source_validation_is_independent_of_macro_validation_percentage(self):
        before = self.split("steps123_to_step4")
        protocol = dict(self.protocol, source_validation={"strategy": "grouped_trials", "fraction": .4})
        after = self.split("steps123_to_step4", protocol)
        self.assertEqual(after["test"], before["test"])
        self.assertGreater(len(after["validation"]), len(before["validation"]))
        self.assertLess(len(after["train"]), len(before["train"]))
        with self.assertRaisesRegex(ValueError, "source_validation"):
            self.split("steps123_to_step4", dict(self.protocol, source_validation={"fraction": 1}))

    def test_cli_reports_and_saves_percentage_plan_for_every_modality(self):
        base = PROJECT_ROOT / "runs" / "semg"
        base.mkdir(parents=True, exist_ok=True)
        for modality in ("imu", "semg", "rgb", "depth"):
            config = {"modality": modality, "model": "invented", "training": {"seed": 42}}
            settings = [config, {"name": "step4_to_step4"}, self.protocol,
                        {"dataset_root": "../dataset", "read_only": True}]
            with tempfile.TemporaryDirectory(prefix="unittest_percentage_", dir=base) as temp:
                output, stdout = Path(temp), io.StringIO()
                with patch.object(cli, "read_config", side_effect=copy.deepcopy(settings)), \
                     patch.object(cli, "create_run_directory", return_value=output), \
                     patch.object(cli, "source_versions", return_value={}), \
                     patch("macro_tools.annotations.load_annotations", return_value=self.records), \
                     contextlib.redirect_stdout(stdout):
                    self.assertEqual(cli.main(modality, ["--protocol", "protocol2_percentages", "--audit-only"]), 0)
                saved = json.loads((output / "config.json").read_text())
                fold = saved["preparation"]["folds"][0]
                self.assertIn("macro_percentage_split", fold["protocol"])
                self.assertEqual({p.name for p in output.iterdir()}, {"config.json"})
                self.assertIn("requested 70/15/15%", stdout.getvalue())
                self.assertIn("actual annotations", stdout.getvalue())

    def test_run_names_identify_percentage_and_split_seed(self):
        name = experiment_run_name("semg", {}, self.protocol)
        self.assertIn("split70-15-15", name)
        self.assertIn("ss42", name)
        self.assertNotEqual(name, experiment_run_name("semg", {}, dict(self.protocol, split_seed=7)))


if __name__ == "__main__":
    unittest.main()
