"""Composite repetition preservation and read-only CSV selection checks."""
import csv
from pathlib import Path
import tempfile
import unittest
from macro_tools.annotations import load_annotations, repetition_metadata, PROJECT_ROOT


class AnnotationTests(unittest.TestCase):
    def test_composite_prefix_groups_substeps_without_losing_raw_text(self):
        record=dict(record_id="r",source_file="file.csv",source_row=2,subject_id="a",session_id="step4",object_id="1",parse_errors=[{"field":"repetition","code":"invalid_number"}])
        one=repetition_metadata(record,"2_1")
        two=repetition_metadata(dict(record,object_id="5"),"2_3")
        self.assertEqual(one["macro_trial_id"],two["macro_trial_id"])
        self.assertEqual(two["repetition_raw"],"2_3")
        self.assertEqual(two["repetition"],2)
        self.assertEqual(two["macro_substep"],3)
        self.assertEqual(two["parse_errors"],[])
        self.assertNotIn("repetition",record)
        with self.assertRaises(ValueError):
            repetition_metadata(record,"2_3","unknown")
        with self.assertRaises(ValueError):
            repetition_metadata(dict(record,session_id="step1"),"2_3")

    def test_modality_pattern_ignores_rgb_annotations_and_source_is_unchanged(self):
        base=PROJECT_ROOT/"runs"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="unittest_annotations_",dir=base) as folder:
            root=Path(folder)
            target=root/"a/annotation"
            target.mkdir(parents=True)
            path=target/"imu_semg_gesture_segments1_EN.csv"
            headers=["session_id","gesture_label","object","repetition","imu_start_s","imu_end_s","imu_start_s_1","imu_end_s_2","semg_start_s","semg_end_s"]
            with path.open("w",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,headers);writer.writeheader()
                writer.writerow(dict(zip(headers,["step4","grasp","1","1_2",1,2,1,2,1,2])))
            before=path.read_bytes()
            (target/"rgb_depth_gesture_segments1_EN.csv").write_text("unrelated",encoding="utf-8")
            rows=load_annotations({"dataset_root":str(root),"read_only":True},"imu")
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]["repetition_raw"],"1_2")
            self.assertEqual(rows[0]["repetition"],1)
            self.assertEqual(rows[0]["macro_substep"],2)
            self.assertEqual(rows[0]["intervals"]["imu_e45a"]["start_s"],1)
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(ValueError):
                load_annotations({"dataset_root":str(root),"read_only":False},"imu")

if __name__=="__main__":
    unittest.main()
