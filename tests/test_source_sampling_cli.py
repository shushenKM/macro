"""Check percentage overrides, saved provenance and descriptive run names."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from macro_tools import cli
from macro_tools.paths import PROJECT_ROOT, create_run_directory
from test_splits import records


class SourceSamplingCLITests(unittest.TestCase):
    def test_override_reaches_splitter_reports_and_all_entrypoints(self):
        base = PROJECT_ROOT / "runs" / "semg"
        base.mkdir(parents=True, exist_ok=True)
        for modality in ("imu", "semg", "rgb", "depth"):
            config = {"modality": modality, "model": "invented", "training": {"seed": 42}}
            experiment = {"name": "steps1234_to_step4", "steps123_train_percent": 100}
            protocol = {"name": "protocol2", "type": "held_out_repetitions", "validation": {"fraction": .2}}
            dataset = {"dataset_root": "../dataset", "read_only": True}
            with tempfile.TemporaryDirectory(prefix="unittest_source_sampling_", dir=base) as temp:
                output, stdout = Path(temp), io.StringIO()
                with patch.object(cli, "read_config", side_effect=copy.deepcopy([config, experiment, protocol, dataset])), \
                     patch.object(cli, "create_run_directory", return_value=output) as directory, \
                     patch.object(cli, "source_versions", return_value={}), \
                     patch("macro_tools.annotations.load_annotations", return_value=records()), \
                     contextlib.redirect_stdout(stdout):
                    self.assertEqual(cli.main(modality, ["--experiment", experiment["name"], "--steps123-percent", "50", "--audit-only"]), 0)
                self.assertEqual(directory.call_args.args[2]["steps123_train_percent"], 50)
                saved = json.loads((output / "config.json").read_text())
                self.assertEqual(saved["experiment"]["steps123_train_percent"], 50)
                fold = saved["preparation"]["folds"][0]
                self.assertEqual(fold["protocol"]["steps123_training_sampling"]["requested_percent"], 50)
                self.assertEqual({p.name for p in output.iterdir()}, {"config.json"})
                self.assertIn("50% requested", stdout.getvalue())

    def test_percentage_in_automatic_run_name(self):
        base = PROJECT_ROOT / "runs" / "semg"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="unittest_source_names_", dir=base) as temp:
            with patch("macro_tools.paths.PROJECT_ROOT", Path(temp)):
                paths = [create_run_directory("semg", {}, {"name": "steps1234_to_step4", "steps123_train_percent": percent},
                                              {"name": "protocol2"}) for percent in (25, 50, 100)]
            self.assertEqual(len(set(paths)), 3)
            for path, percent in zip(paths, (25, 50, 100)):
                self.assertIn("src{}pct".format(percent), path.name)


if __name__ == "__main__":
    unittest.main()
