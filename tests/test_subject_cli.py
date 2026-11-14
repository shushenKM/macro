"""Shared modality flags and final averages from independent training folds."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from macro_tools import cli
from macro_tools.metrics import aggregate_fold_metrics, format_fold_summary
from macro_tools.paths import PROJECT_ROOT, experiment_run_name
from test_subject_protocols import eight_subject_records, subject_config


class SubjectCLITests(unittest.TestCase):
    def config_values(self, modality):
        return [dict(modality=modality, model="invented", training={"seed": 42}),
                {"name": "step4_to_step4"},
                subject_config(subject_counts=dict(train=None, validation=1, test=1)),
                dict(dataset_root="../dataset", read_only=True)]

    def test_fixed_pair_flags_and_provenance_in_every_modality(self):
        base = PROJECT_ROOT / "runs" / "imu"
        base.mkdir(parents=True, exist_ok=True)
        for modality in ("imu", "semg", "rgb", "depth"):
            with tempfile.TemporaryDirectory(prefix="unittest_subject_cli_", dir=base) as temporary:
                output, stdout = Path(temporary), io.StringIO()
                with patch.object(cli, "read_config", side_effect=self.config_values(modality)), \
                     patch.object(cli, "create_run_directory", return_value=output), \
                     patch.object(cli, "source_versions", return_value={}), \
                     patch("macro_tools.annotations.load_annotations", return_value=eight_subject_records()), \
                     contextlib.redirect_stdout(stdout):
                    status = cli.main(modality, ["--protocol", "protocol1", "--test-subjects", "a", "b",
                                                "--validation-subjects", "d", "e", "--train-subject-count", "4", "--audit-only"])
                self.assertEqual(status, 0)
                saved = json.loads((output / "config.json").read_text())
                self.assertEqual(saved["protocol"]["subject_counts"], dict(train=4, validation=2, test=2))
                folds = saved["preparation"]["folds"]
                self.assertEqual(len(folds), 1)
                self.assertEqual(folds[0]["protocol"]["test_subject_ids"], ["a", "b"])
                self.assertEqual({p.name for p in output.iterdir()}, {"config.json"})
                self.assertIn("participants train/validation/test 4/2/2", stdout.getvalue())

    def test_all_eight_train_calls_then_saved_and_printed_average(self):
        base = PROJECT_ROOT / "runs" / "imu"
        base.mkdir(parents=True, exist_ok=True)
        def prepare(_records, folds, _dataset, _config):
            return dict(folds=copy.deepcopy(folds), label_mapping={"grasp": 0, "insert": 1}, issues=[])
        def train_fold(fold, *_args):
            index = ord(fold["metadata"]["test_subject_ids"][0]) - ord("a")
            value = index / 10
            return dict(name=fold["name"], test_metrics=dict(accuracy=value, macro_f1=value, weighted_f1=value, sample_count=len(fold["test"])))
        with tempfile.TemporaryDirectory(prefix="unittest_subject_average_", dir=base) as temporary:
            output, stdout = Path(temporary), io.StringIO()
            with patch.object(cli, "read_config", side_effect=self.config_values("imu")), \
                 patch.object(cli, "create_run_directory", return_value=output), \
                 patch.object(cli, "source_versions", return_value={}), \
                 patch("macro_tools.annotations.load_annotations", return_value=eight_subject_records()), \
                 patch("macro_tools.adapters.prepare_dataset", side_effect=prepare), \
                 patch("macro_tools.training.save_prepared_fold"), \
                 patch("macro_tools.training.train_fold", side_effect=train_fold) as trainer, \
                 patch("macro_tools.plots.save_training_report"), contextlib.redirect_stdout(stdout):
                self.assertEqual(cli.main("imu", ["--protocol", "protocol1", "--train-subject-count", "6",
                                                   "--validation-subject-count", "1", "--test-subject-count", "1"]), 0)
            self.assertEqual(trainer.call_count, 8)
            self.assertEqual(len({call.args[3] for call in trainer.call_args_list}), 8)
            aggregate = json.loads((output / "aggregate_metrics.json").read_text())
            self.assertEqual(aggregate["fold_count"], 8)
            self.assertAlmostEqual(aggregate["accuracy"]["mean"], .35)
            self.assertAlmostEqual(aggregate["accuracy"]["std"], .229128784747792)
            self.assertIn("35.00 +/- 22.91", stdout.getvalue())
            self.assertIn("Training fold 8/8", stdout.getvalue())
            self.assertIn("35.00 +/- 22.91", (output / "fold_summary.txt").read_text())

    def test_invalid_options_fail_before_annotation_reading(self):
        for arguments in (["--test-subject-count", "2"],
                          ["--test-subjects", "a", "b", "--test-subject-count", "1"],
                          ["--train-subject-count", "0"]):
            with patch.object(cli, "read_config", side_effect=self.config_values("imu")), \
                 patch("macro_tools.annotations.load_annotations") as annotations, \
                 patch.object(cli, "create_run_directory") as output, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main("imu", ["--protocol", "protocol1"] + arguments), 2)
                annotations.assert_not_called()
                output.assert_not_called()
        values = self.config_values("imu")
        values[2] = dict(name="protocol2", type="held_out_repetitions")
        with patch.object(cli, "read_config", side_effect=values), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main("imu", ["--test-subject-count", "1"]), 2)

    def test_names_distinguish_counts_and_fixed_test_groups(self):
        protocols = [subject_config(),
                     subject_config(subject_counts=dict(train=5, validation=2, test=1)),
                     subject_config(test_subject_ids=["a", "b"]),
                     subject_config(test_subject_ids=["c", "d"])]
        names = [experiment_run_name("imu", {}, protocol) for protocol in protocols]
        self.assertEqual(len(set(names)), 4)
        self.assertIn("subjectsauto-auto-1_loso", names[0])
        self.assertIn("test-a-b", names[2])

    def test_equal_fold_mean_is_not_pooled_sample_accuracy(self):
        metrics = [dict(accuracy=.5, macro_f1=.4, weighted_f1=.45, sample_count=100),
                   dict(accuracy=1., macro_f1=.9, weighted_f1=.95, sample_count=2)]
        aggregate = aggregate_fold_metrics(metrics)
        self.assertEqual(aggregate["accuracy"], dict(mean=.75, std=.25))
        results = [dict(name=str(i), test_metrics=m) for i, m in enumerate(metrics)]
        self.assertIn("75.00 +/- 25.00", format_fold_summary(results, aggregate))
        self.assertIn("One completed fold", format_fold_summary(results[:1]))


if __name__ == "__main__":
    unittest.main()
