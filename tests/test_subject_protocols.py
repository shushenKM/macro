"""Participant counts, complete LOSO coverage and fixed multi-subject holdouts."""
import copy
import unittest

from macro_tools.splits import build_splits
from macro_tools.subject_splits import normalize_subject_protocol, participant_partitions
from test_splits import records


def eight_subject_records():
    rows = records()
    additional = []
    for row in rows:
        item = copy.deepcopy(row)
        subject = chr(ord(row["subject_id"]) + 4)
        item["subject_id"] = subject
        item["record_id"] = subject + ":" + row["record_id"]
        additional.append(item)
    return rows + additional


def subject_config(**kwargs):
    return dict(name="protocol1", type="subject_holdout", **kwargs)


class SubjectProtocolTests(unittest.TestCase):
    def setUp(self):
        self.records = eight_subject_records()
        self.subjects = list("abcdefgh")

    def test_default_loso_all_eight_once_with_no_subject_leakage(self):
        before = copy.deepcopy(self.records)
        sets = []
        for experiment in ("step4_to_step4", "steps1234_to_step4", "steps123_to_step4"):
            folds = build_splits(self.records, experiment, subject_config())
            self.assertEqual(len(folds), 8)
            self.assertEqual([f["metadata"]["test_subject_ids"] for f in folds], [[s] for s in self.subjects])
            sets.append([{r["record_id"] for r in f["test"]} for f in folds])
            for index, fold in enumerate(folds):
                people = {part: {r["subject_id"] for r in fold[part]} for part in ("train", "validation", "test")}
                self.assertEqual(fold["metadata"]["subject_counts"], dict(train=6, validation=1, test=1))
                self.assertFalse(people["train"] & (people["validation"] | people["test"]))
                self.assertFalse(people["validation"] & people["test"])
                self.assertEqual(people["validation"], {self.subjects[(index + 1) % 8]})
                expected = {"step1"} if experiment == "steps123_to_step4" else {"step4"}
                self.assertEqual({r["session_id"] for r in fold["validation"]}, expected)
        self.assertEqual(sets[0], sets[1])
        self.assertEqual(sets[0], sets[2])
        self.assertEqual(self.records, before)

    def test_fixed_pair_single_fold_and_explicit_validation(self):
        config = subject_config(subject_counts=dict(train=4, validation=2, test=2),
                                test_subject_ids=["h", "b"], validation_subject_ids=["a", "e"])
        for experiment in ("step4_to_step4", "steps1234_to_step4", "steps123_to_step4"):
            folds = build_splits(self.records, experiment, config)
            self.assertEqual(len(folds), 1)
            fold = folds[0]
            self.assertEqual(fold["metadata"]["test_subject_ids"], ["b", "h"])
            self.assertEqual(fold["metadata"]["validation_subject_ids"], ["a", "e"])
            self.assertEqual({r["subject_id"] for r in fold["train"]}, set("cdfg"))
            self.assertEqual({r["subject_id"] for r in fold["test"]}, set("bh"))
            self.assertEqual(fold["metadata"]["subject_selection"], "fixed")

    def test_train_count_excludes_unused_people_from_every_context(self):
        config = subject_config(subject_counts=dict(train=3, validation=2, test=1))
        for fold in build_splits(self.records, "steps1234_to_step4", config):
            meta = fold["metadata"]
            self.assertEqual(meta["subject_counts"], dict(train=3, validation=2, test=1))
            self.assertEqual(len(meta["excluded_subject_ids"]), 2)
            unused = set(meta["excluded_subject_ids"])
            self.assertTrue(all(r["subject_id"] not in unused for p in ("train", "validation", "test") for r in fold[p]))
            self.assertTrue(any(r["exclusion_reason"] == "participant_not_selected" for r in fold["excluded"]))

    def test_no_validation_and_order_independence(self):
        config = subject_config(subject_counts=dict(train=None, validation=0, test=1))
        folds = build_splits(self.records, "step4_to_step4", config)
        reverse = build_splits(list(reversed(self.records)), "step4_to_step4", config)
        for original, reordered in zip(folds, reverse):
            self.assertEqual(original["metadata"], reordered["metadata"])
            self.assertEqual(original["metadata"]["subject_counts"], dict(train=7, validation=0, test=1))
            self.assertEqual(original["validation"], [])
            self.assertTrue(any(i["code"] == "no_validation_split" for i in original["issues"]))

    def test_legacy_singular_config_keeps_original_assignments(self):
        legacy = dict(name="protocol1", type="leave_one_subject_out", test_subject_id="a",
                      validation_subject_id="c", validation={"strategy": "next_subject"})
        selection = participant_partitions(self.subjects, legacy)[0]
        self.assertEqual(selection["test"], ["a"])
        self.assertEqual(selection["validation"], ["c"])
        self.assertEqual(selection["train"], list("bdefgh"))
        legacy["validation"] = {"strategy": "none"}
        legacy["validation_subject_id"] = None
        self.assertEqual(len(participant_partitions(self.subjects, legacy)[0]["train"]), 7)

    def test_invalid_or_ambiguous_counts_and_ids_fail(self):
        cases = [
            dict(subject_counts=dict(test=2)),
            dict(subject_counts=dict(train=0)),
            dict(subject_counts=dict(validation=-1)),
            dict(subject_counts=dict(test=True)),
            dict(subject_counts=dict(train=2.5)),
            dict(subject_counts=dict(test=1), test_subject_ids=["a", "b"]),
            dict(test_subject_ids=["a", "a"]),
            dict(test_subject_ids=[]),
            dict(test_subject_ids="a"),
            dict(test_subject_ids=["a"], validation_subject_ids=["a"]),
            dict(validation_subject_ids=["a"]),
            dict(test_subject_id="a", test_subject_ids=["b"]),
            dict(subject_counts=dict(validation=1), validation={"strategy": "none"}),
            dict(subject_counts=dict(training=6)),
        ]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                normalize_subject_protocol(subject_config(**settings))
        for settings in (dict(subject_counts=dict(train=7, validation=1, test=1)),
                         dict(test_subject_ids=["unknown"]),
                         dict(test_subject_ids=["a"], validation_subject_ids=["unknown"])):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                participant_partitions(self.subjects, subject_config(**settings))

    def test_selected_source_participant_requires_source_annotations(self):
        rows = [r for r in self.records if not (r["subject_id"] == "c" and r["session_id"] != "step4")]
        with self.assertRaisesRegex(ValueError, "no eligible annotations"):
            build_splits(rows, "steps123_to_step4", subject_config(test_subject_ids=["a"]))


if __name__ == "__main__":
    unittest.main()
