"""Macro TCN integration on invented arrays only; preserve training settings."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
# Discovery imports all test modules before execution. Establish the CUDA
# workspace before earlier GPU tests initialize cuBLAS in this shared process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from macro_tools import cli, training
from macro_tools.adapters.models import build_model
from macro_tools.paths import PROJECT_ROOT, read_config


class MacroTCNTests(unittest.TestCase):
    def test_tcn_configs_differ_only_in_model_fields(self):
        for modality in ("imu", "semg"):
            base = read_config("configs/" + modality + ".json")
            tcn = read_config("configs/" + modality + "_tcn.json")
            self.assertEqual(tcn["model"], "TCN")
            for config in (base, tcn):
                for key in ("model", "model_parameters", "upstream_project"):
                    config.pop(key, None)
            self.assertEqual(base, tcn)

    def test_cli_only_changes_architecture_across_experiments_and_protocols(self):
        for modality in ("imu", "semg"):
            base = read_config("configs/" + modality + ".json")
            for experiment in ("step4_to_step4", "steps1234_to_step4", "steps123_to_step4"):
                for protocol in ("protocol1", "protocol2", "protocol2_percentages"):
                    folder = PROJECT_ROOT / "runs" / modality
                    folder.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix="tcn_cli_test_", dir=folder) as scratch:
                        output = Path(scratch)
                        with patch.object(cli, "create_run_directory", return_value=output), \
                             patch.object(cli, "source_versions", return_value={}), \
                             patch("macro_tools.annotations.load_annotations", return_value=[]), \
                             patch("macro_tools.splits.build_splits", return_value=[]), \
                             contextlib.redirect_stdout(io.StringIO()):
                            status = cli.main(modality, ["--model", "tcn", "--experiment", experiment,
                                                        "--protocol", protocol, "--audit-only"])
                        self.assertEqual(status, 0)
                        saved = json.loads((output / "config.json").read_text())
                        effective = saved["config"]
                        self.assertEqual(effective["model"], "TCN")
                        for key, value in base.items():
                            if key not in ("model", "model_parameters", "upstream_project"):
                                self.assertEqual(effective[key], value, key)
                        self.assertEqual(saved["experiment"], read_config("configs/experiments/" + experiment + ".json"))
                        expected_protocol = read_config("configs/protocols/" + protocol + ".json")
                        if protocol == "protocol1":
                            from macro_tools.subject_splits import normalize_subject_protocol
                            expected_protocol = normalize_subject_protocol(expected_protocol)
                        self.assertEqual(saved["protocol"], expected_protocol)

    def test_builders_support_single_and_dual_sensor_grid_inputs(self):
        torch.set_num_threads(1)
        for modality in ("imu", "semg"):
            for count in (1, 2):
                config = read_config("configs/" + modality + "_tcn.json")
                config["input"]["num_points"] = 17
                key, names = (("sensor_ids", ["E45A", "E477"]) if modality == "imu"
                              else ("grid_ids", ["grid1", "grid2"]))
                config[key] = names[-count:]
                model = build_model(config, 3)
                shape = (2, 1, 9*count, 17) if modality == "imu" else (2, 8, 17, 8*count)
                logits = model(torch.randn(shape))
                self.assertEqual(tuple(logits.shape), (2, 3))
                logits.sum().backward()
                self.assertEqual(type(model.model).__name__, "SignalTCN")
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_real_macro_trainer_and_checkpoint_reload_cpu_and_cuda(self):
        rng = np.random.default_rng(42)
        labels = {"approach": 0, "grasp": 1, "release": 2}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for modality in ("imu", "semg"):
                with self.subTest(device=device, modality=modality):
                    config = read_config("configs/" + modality + "_tcn.json")
                    # Use the full default architecture and current AMP/scheduler setup.
                    config["training"].update(epochs=1, batch_size=3, device=device)
                    config["synthetic"] = True
                    points = config["input"]["num_points"]
                    shape = (1, 18, points) if modality == "imu" else (8, points, 16)
                    splits = {part: dict(x=rng.normal(size=(count, *shape)).astype(np.float32),
                                         y=np.arange(count, dtype=np.int64) % 3,
                                         metadata=[dict(sample_id=part+str(i), parent_record_id=part+str(i))
                                                   for i in range(count)])
                              for part, count in (("train", 6), ("validation", 3), ("test", 3))}
                    fold = dict(name="tcn_synthetic", splits=splits, normalization={"fit_partition": "train"})
                    folder = PROJECT_ROOT / "runs" / modality
                    folder.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix="tcn_training_test_", dir=folder) as scratch:
                        with patch.object(training, "evaluate", wraps=training.evaluate) as evaluate, \
                             patch("macro_tools.plots.save_training_report"), contextlib.redirect_stdout(io.StringIO()):
                            result = training.train_fold(fold, labels, config, scratch, "unused_synthetic")
                        self.assertEqual(evaluate.call_count, 2)  # validation then test
                        self.assertEqual(result["model"], "TCN")
                        payload = torch.load(Path(scratch) / result["selected_checkpoint"], weights_only=True, map_location=device)
                        self.assertEqual(payload["config"], config)
                        restored = build_model(payload["config"], 3).to(device).eval()
                        restored.load_state_dict(payload["model_state_dict"])
                        amp = device == "cuda" and config["training"].get("amp", False)
                        with torch.inference_mode(), torch.autocast(device_type=device, enabled=amp):
                            predictions = restored(torch.from_numpy(splits["test"]["x"]).to(device)).argmax(1).cpu().tolist()
                        with (Path(scratch) / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                            saved = list(csv.DictReader(stream))
                        self.assertEqual(predictions, [int(row["prediction"]) for row in saved])
                        self.assertTrue((Path(scratch) / "history.csv").is_file())
                        self.assertEqual(result["training_details"]["model"]["optimizer"], "AdamW")


if __name__ == "__main__":
    unittest.main()
