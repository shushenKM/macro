"""Guard the finite-input FP16 overflow seen in macro sEMG validation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from macro_tools import cli, training
from macro_tools.paths import PROJECT_ROOT, read_config


class PrecisionTests(unittest.TestCase):
    def test_semg_defaults_to_fp32_and_explicit_amp_is_supported(self):
        self.assertFalse(training._settings({"modality": "semg"})["amp"])
        self.assertFalse(training._settings(read_config("configs/semg.json"))["amp"])
        self.assertTrue(training._settings({"modality": "semg", "training": {"amp": True}})["amp"])
        self.assertTrue(training._settings({"modality": "rgb"})["amp"])

    def test_no_amp_cli_override_is_saved_even_when_config_enables_amp(self):
        base = PROJECT_ROOT / "runs" / "semg"
        base.mkdir(parents=True, exist_ok=True)
        config = {"modality": "semg", "model": "VisionTransformer", "training": {"amp": True}}
        dataset = {"dataset_root": "../dataset", "read_only": True}
        fold = {"name": "invented", "train": [], "validation": [], "test": [], "issues": []}
        with tempfile.TemporaryDirectory(prefix="unittest_precision_", dir=base) as temporary:
            run = Path(temporary)
            with patch.object(cli, "read_config", side_effect=[config, {"name": "experiment"}, {"name": "protocol"}, dataset]), \
                 patch.object(cli, "create_run_directory", return_value=run), \
                 patch.object(cli, "source_versions", return_value={}), \
                 patch("macro_tools.annotations.load_annotations", return_value=[]), \
                 patch("macro_tools.splits.build_splits", return_value=[fold]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main("semg", ["--no-amp", "--audit-only"]), 0)
            saved = json.loads((run / "config.json").read_text())
            self.assertIs(saved["config"]["training"]["amp"], False)

    def test_numerical_failure_is_reported_without_uncaught_traceback(self):
        stderr = io.StringIO()
        with patch.object(cli, "train", side_effect=FloatingPointError("non-finite logits")), \
             contextlib.redirect_stderr(stderr):
            self.assertEqual(cli.main("semg", []), 2)
        self.assertIn("Macro semg pipeline stopped: non-finite logits", stderr.getvalue())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required to reproduce FP16 overflow")
    def test_large_finite_semg_input_evaluates_in_fp32_on_cuda(self):
        config = read_config("configs/semg.json")
        torch.manual_seed(42)
        model = training._make_model(config, 2).cuda()
        inputs = torch.randn(2, 8, 128, 16)
        inputs[1, 2, 3, 4] = 289240.40625
        original = inputs.clone()
        loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 1])), batch_size=2)
        labels, device = {"a": 0, "b": 1}, torch.device("cuda")
        with self.assertRaises(FloatingPointError):
            training.evaluate(model, loader, labels, device, amp=True)
        metrics, predictions = training.evaluate(model, loader, labels, device,
                                                  amp=training._settings(config)["amp"])
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual(len(predictions), 2)
        torch.testing.assert_close(inputs, original)


if __name__ == "__main__":
    unittest.main()
