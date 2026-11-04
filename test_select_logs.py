"""Small fixtures for label selection, provenance separation, and unknown labels."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import select_logs as s


def document(log_id, codes):
    return {"filename": log_id, "drivableAreas": {}, "pedCrossings": {},
            "laneSegments": {"ls-1": {"changes": codes, "modified": False,
                                       "old": {"id": "ls-1"}, "new": None}}}


class SelectionTests(unittest.TestCase):
    def test_null_new_and_modified_are_not_labels(self):
        self.assertEqual(s.parse_changes(document("x", [0]), "x"), [])
        # A changed element may also have new=null; do not infer a direction here.
        self.assertEqual(len(s.parse_changes(document("x", [2]), "x")), 1)

    def test_missing_or_malformed_label_is_not_no_change(self):
        for codes in ([], None, [0, 3], [True]):
            with self.assertRaises(ValueError):
                s.parse_changes(document("x", codes), "x")

    def test_real_constructed_and_unknown_are_separate(self):
        self.assertEqual(s.classify("train", "train", None, 5), "constructed_pool")
        self.assertEqual(s.classify("test", None, None, None), "unlabeled")
        self.assertEqual(s.classify("val", "test", 1, None), "review")

    def test_union_selection_and_disagreement_in_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tbv, at, metadata = [root / x for x in ("tbv", "at", "metadata")]
            metadata.mkdir()
            (at / "test").mkdir(parents=True)
            (at / "train+val").mkdir()
            ids = [f"scene{i}__Summer_2020" for i in range(6)]
            for log_id in ids:
                log = tbv / "TbV_v1.0_shard0" / log_id
                for part in ("map", "calibration", "sensors/lidar"):
                    (log / part).mkdir(parents=True, exist_ok=True)
                for camera in s.CAMERAS:
                    (log / "sensors/cameras" / camera).mkdir(parents=True)
                for file in ("city_SE3_egovehicle.feather", "calibration/intrinsics.feather",
                             "calibration/egovehicle_SE3_sensor.feather", "map/log_map_archive_example.json"):
                    (log / file).touch()
            split = {"val": ids[:4], "train": [ids[4]], "test": [ids[5]]}
            at_split = {"test": {str(i): v for i, v in enumerate(ids[:4])},
                        "train": {"0": ids[4]}, "val": {}}
            event = {"change_type": "lane_geometry_change", "supercategory": "lane_geometry_change"}
            annotations = [{"log_id": v, "events": [event] if i in (0, 2) else []}
                           for i, v in enumerate(ids[:4])]
            for name, value in (("tbv_splits.json", split), ("argotweak_splits.json", at_split),
                                ("tbv_val_split_annotations.json", annotations), ("sources.json", {})):
                (metadata / name).write_text(json.dumps(value))
            for i, log_id in enumerate(ids[:5]):
                folder = "test" if i < 4 else "train+val"
                (at / folder / (log_id + ".json")).write_text(json.dumps(
                    document(log_id, [3] if i in (0, 1, 4) else [0])))
            args = argparse.Namespace(tbv_root=tbv, argotweak_root=at, metadata=metadata,
                                      out=root / "out", scan_constructed=False)
            with contextlib.redirect_stdout(io.StringIO()):
                s.run(args)
            report = s.read_json(args.out / "report.json")
            self.assertEqual(report["buckets"], {"real_change": 3, "real_no_change_control": 1,
                                                "constructed_pool": 1, "unlabeled": 1})
            self.assertEqual(report["output_rows"]["review_logs.jsonl"], 2)
            train = json.loads((args.out / "constructed_pool.jsonl").read_text())
            self.assertIsNone(train["argotweak_changed_element_count"])
            self.assertFalse(train["argotweak_scanned"])
            with self.assertRaises(ValueError):
                s.run(args)  # refuse to overwrite an existing result

    def test_duplicate_logs_fail_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for shard in (0, 1):
                (root / f"TbV_v1.0_shard{shard}" / "same__Summer_2020").mkdir(parents=True)
            with self.assertRaises(ValueError):
                s.index_logs(root)


if __name__ == "__main__":
    unittest.main()
