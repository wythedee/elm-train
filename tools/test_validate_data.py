#!/usr/bin/env python3
"""Synthetic manifest identity tests; no model, optimizer, or EEG data."""
import json
from pathlib import Path
import tempfile
import unittest

from validate_data import REPO, audit, canonical_subject


class IdentityTests(unittest.TestCase):
    def test_known_aliases(self):
        self.assertEqual(canonical_subject("MUMTAZ_MDD", "H_S02"),
                         canonical_subject("NEMAR_MDD", "sub-HS2"))
        self.assertEqual(canonical_subject("MUMTAZ_MDD", "MDD_S03"),
                         canonical_subject("NEMAR_MDD", "sub-MDDS3"))
        with self.assertRaises(ValueError):
            canonical_subject("NEMAR_MDD", "unknown_subject")
        self.assertEqual(canonical_subject("CHB_MIT", "chb01"),
                         canonical_subject("CHB_MIT", "chb21"))

    def fixture(self, root, leak=False, duplicate=False):
        for dataset in ("MUMTAZ_MDD", "NEMAR_MDD"):
            target = root / dataset
            target.mkdir()
            for i, split in enumerate(("train", "val", "test"), 1):
                n = 1 if leak and dataset == "NEMAR_MDD" and split == "val" else i
                row = {
                    "dataset": dataset, "id": f"{dataset}_{split}",
                    "subject": f"H_S{n:02d}" if dataset == "MUMTAZ_MDD" else f"sub-HS{n}",
                    "label": "healthy_control", "sampling_rate": 200,
                    "input_shape": [2, 1000], "channels": ["Fp1", "Fp2"],
                    "canonical_channel_indices": [6, 7], "patch_count": 5,
                    "patch_size": 200, "lmdb_path": "prepared/data.lmdb",
                    "lmdb_key": f"{dataset}_{split}",
                }
                lines = [json.dumps(row)]
                if duplicate and split == "train":
                    lines.append(json.dumps(row))
                (target / (split + ".jsonl")).write_text("\n".join(lines) + "\n")

    def test_cross_source_split_leak_is_rejected(self):
        cache = REPO / "cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            self.fixture(Path(temp), leak=True)
            result = audit(temp, ["MUMTAZ_MDD", "NEMAR_MDD"])
            self.assertEqual(result["status"], "failed")
            self.assertGreater(result["cross_split_subject_overlap_count"], 0)

    def test_same_split_alias_is_declared_not_false_cross_split(self):
        cache = REPO / "cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            self.fixture(Path(temp))
            result = audit(temp, ["MUMTAZ_MDD", "NEMAR_MDD"])
            self.assertEqual(result["status"], "passed_manifest_checks")
            self.assertEqual(result["cross_split_subject_overlap_count"], 0)
            self.assertEqual(result["warnings"][0]["people"], 3)

    def test_tolerance_cannot_hide_duplicate_ids(self):
        cache = REPO / "cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            self.fixture(Path(temp), duplicate=True)
            result = audit(temp, ["MUMTAZ_MDD", "NEMAR_MDD"], 0.1)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any(e["kind"] == "duplicate_id" for e in result["errors"]))

    def test_declared_hebin_lazy_h5_is_not_treated_as_lmdb(self):
        cache = REPO / "cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            root = Path(temp)
            dataset = "HEBIN2021_NATIVE200_LR"
            target = root / dataset
            target.mkdir()
            for i, split in enumerate(("train", "val", "test")):
                row = {"id": f"hebin_{i}", "dataset": dataset, "subject": f"S{i}",
                       "label": "left_hand", "h5_path": "/external/MI.h5",
                       "trial_idx": 0, "source_sampling_rate": 250,
                       "sampling_rate": 200, "resample_on_load": True,
                       "window_start": 125, "window_length": 1000,
                       "input_shape": [75, 800],
                       "canonical_channel_indices": list(range(75))}
                (target / (split + ".jsonl")).write_text(json.dumps(row) + "\n")
            self.assertEqual(audit(root, [dataset])["status"], "passed_manifest_checks")
            row["window_start"] = 0
            (target / "test.jsonl").write_text(json.dumps(row) + "\n")
            self.assertEqual(audit(root, [dataset], 0.99)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
