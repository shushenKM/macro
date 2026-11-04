"""Whole-macro-trial isolation and source-context evaluation tests."""
import copy
import unittest
from macro_tools.splits import build_splits, trial_key


def records():
    result=[]
    for subject in ("a","b","c","d"):
        for session in ("step1","step4"):
            for obj in ("1","2"):
                for rep in (1,2,3):
                    for substep in (1,2):
                        for gesture in ("grasp","insert"):
                            result.append(dict(record_id="/".join(map(str,(subject,session,obj,rep,substep,gesture))),
                                subject_id=subject,session_id=session,object_id=obj,repetition=rep,
                                macro_trial_id="/".join((subject,session,"assembly"+str(rep))) if session=="step4" else None,
                                gesture_label=gesture,intervals={}))
    result.append(dict(record_id="source_only",subject_id="a",session_id="step1",object_id="1",repetition=1,gesture_label="place",intervals={}))
    return result


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.records=records()
        self.p2=dict(name="protocol2",type="held_out_macro_repetitions",train_repetitions=[1,2],test_repetitions=[3],validation={"fraction":.2})
        self.p1=dict(name="protocol1",type="leave_one_subject_out",test_subject_id="a")

    def test_three_scenarios_share_identical_target_test_set_and_pure_transfer_validation(self):
        before=copy.deepcopy(self.records)
        folds={name:build_splits(self.records,{"name":name},self.p2)[0] for name in ("step4_to_step4","steps1234_to_step4","steps123_to_step4")}
        expected={r["record_id"] for r in folds["step4_to_step4"]["test"]}
        for fold in folds.values():
            self.assertEqual({r["record_id"] for r in fold["test"]},expected)
            self.assertTrue(all(r["session_id"]=="step4" and r["repetition"]==3 for r in fold["test"]))
            assignments={}
            for part in ("train","validation","test"):
                for record in fold[part]:
                    key=trial_key(record)
                    self.assertIn(assignments.get(key,part),[part])
                    assignments[key]=part
        self.assertEqual({r["record_id"] for r in folds["step4_to_step4"]["validation"]},
                         {r["record_id"] for r in folds["steps1234_to_step4"]["validation"]})
        self.assertTrue(all(r["session_id"]!="step4" for part in ("train","validation") for r in folds["steps123_to_step4"][part]))
        self.assertEqual(self.records,before)
        self.assertEqual(folds["steps1234_to_step4"]["metadata"]["target_gesture_labels"],["grasp","insert"])
        excluded={r["record_id"]:r["exclusion_reason"] for r in folds["steps1234_to_step4"]["excluded"]}
        self.assertEqual(excluded["source_only"],"gesture_outside_target_vocabulary")

    def test_unseen_participant_is_excluded_from_every_training_context(self):
        for name in ("step4_to_step4","steps1234_to_step4","steps123_to_step4"):
            fold=build_splits(self.records,name,self.p1)[0]
            self.assertEqual({r["subject_id"] for r in fold["train"]},{"c","d"})
            self.assertEqual({r["subject_id"] for r in fold["validation"]},{"b"})
            self.assertEqual({r["subject_id"] for r in fold["test"]},{"a"})
            self.assertEqual({r["session_id"] for r in fold["test"]},{"step4"})

    def test_overlap_is_warning_while_duplicate_identity_is_error(self):
        rows=copy.deepcopy(self.records)
        for record in rows:
            record["intervals"]={"imu_e45a":{"start_s":1,"end_s":2}}
        fold=build_splits(rows,"step4_to_step4",self.p2)[0]
        self.assertTrue(any(i["code"]=="overlapping_intervals" and i["severity"]=="warning" for i in fold["issues"]))
        with self.assertRaisesRegex(ValueError,"Duplicate source"):
            build_splits(rows+[rows[0]],"step4_to_step4",self.p2)

    def test_bad_protocol_and_target_class_coverage_fail(self):
        with self.assertRaisesRegex(ValueError,"disjoint"):
            build_splits(self.records,"step4_to_step4",dict(self.p2,test_repetitions=[2]))
        with self.assertRaisesRegex(ValueError,"Training is missing"):
            build_splits(self.records,{"name":"step4_to_step4","target_gesture_labels":["grasp","insert","place"]},self.p2)
        with self.assertRaisesRegex(ValueError,"No step4"):
            build_splits([r for r in self.records if r["session_id"]!="step4"],"steps123_to_step4",self.p2)

if __name__=="__main__":
    unittest.main()
