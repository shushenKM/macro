"""Check macro adapters without source writes or large training jobs."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from macro_tools.adapters import data

ROOT = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def test_step4_imu_resolution_and_escape_protection(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="adapter_test_", dir=ROOT / "runs") as temp:
            root = Path(temp)
            folder = root / "subject" / "imu"
            folder.mkdir(parents=True)
            path = folder / "session4_Shimmer_E45A_Calibrated_PC.csv"
            path.write_text("invented", encoding="utf-8")
            self.assertEqual(data.resolve_imu_recording(root, "subject", "step4", "E45A"), path)
            with self.assertRaises(ValueError):
                data.resolve_imu_recording(root, "../escape", "step4", "E45A")
            with self.assertRaises(ValueError):
                data.resolve_imu_recording(root, "subject", "step4", "E45A", overrides={"subject/step4/E45A": "../../escape.csv"})

    def test_video_override_is_portable_and_cannot_escape_dataset(self):
        root = ROOT / "runs" / "invented_dataset"
        config = {"session_mapping": {"step4": "session4"},
                  "video_recording_overrides": {"person/step4/above": "person/rgbd/original name/above"}}
        result = data.resolve_video_folder(root, config, "person", "step4", "above")
        self.assertEqual(result, root / "person/rgbd/original name/above")
        config["video_recording_overrides"]["person/step4/above"] = "../../escape"
        with self.assertRaises(ValueError):
            data.resolve_video_folder(root, config, "person", "step4", "above")

    def test_complete_step4_screw_imu_interval_is_one_example(self):
        record = dict(record_id="one", subject_id="person", session_id="step4", gesture_label="screw",
                      macro_trial_id="person/step4/task/1", intervals={"imu_e45a": dict(start_s=10., end_s=10.5)})
        prepared = {"E45A": SimpleNamespace(timestamps_s=np.arange(5)/10,
                                             features=np.arange(45).reshape(5, 9))}
        examples = data._full_imu_example(record, prepared, ["E45A"], 16)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0][0].shape, (1, 9, 16))
        self.assertEqual(examples[0][1]["gesture_label"], "screw")
        self.assertEqual(examples[0][1]["macro_trial_id"], record["macro_trial_id"])
        self.assertEqual(examples[0][1]["sensor_windows"]["E45A"], {"start_s": 10., "end_s": 10.5})

    def test_only_assigned_records_are_loaded_and_scaler_uses_training_only(self):
        train = dict(record_id="train", subject_id="a", session_id="step4", gesture_label="grasp")
        test = dict(record_id="test", subject_id="b", session_id="step4", gesture_label="grasp")
        excluded = dict(record_id="excluded", subject_id="c", session_id="step1", gesture_label="place")
        fold = dict(name="fold", train=[train], validation=[], test=[test], excluded=[excluded])
        self.assertEqual(data._assigned([fold]), [train, test])
        windows = {r["record_id"]: [(np.full((1, 9, 16), value, dtype=np.float32), r)]
                   for r, value in ((train, 1), (test, 101))}
        with patch.object(data, "_signal_examples", return_value=(windows, (1, 9, 16), (0, 1, 3))):
            result = data._signal_folds([train, test], [fold], {}, {"modality": "imu"}, {"grasp": 0}, [], [])
        self.assertEqual(result[0]["normalization"]["mean"], [1.] * 9)
        np.testing.assert_array_equal(result[0]["splits"]["train"]["x"], 0)
        np.testing.assert_array_equal(result[0]["splits"]["test"]["x"], 100)
        self.assertEqual(result[0]["splits"]["validation"]["x"].shape, (0, 1, 9, 16))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy needed by sEMG preprocessing")
    def test_semg_step4_short_screw_full_interval_and_grid_shape(self):
        from macro_tools.sources import activate_source
        activate_source("semg")
        from semg.dataset import _examples
        record = dict(record_id="r", subject_id="s", session_id="step4", object_id="1", repetition=1,
                      gesture_label="screw", source_file="synthetic", source_row=1,
                      intervals={"semg": dict(start_s=10., end_s=10.5)})
        values = SimpleNamespace(timestamps_s=np.arange(50)/100, features=np.ones((50, 64)), sample_rate_hz=100)
        examples = _examples(record, values, ["grid2"], {"grid2": list(range(65, 129))}, 16, .5, 1, "error")
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0][0].shape, (8, 16, 8))
        self.assertEqual(examples[0][1]["duration_s"], .5)


if __name__ == "__main__":
    unittest.main()
