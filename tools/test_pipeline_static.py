#!/usr/bin/env python3
"""Laptop-safe code/task tests. No Torch/model import or optimizer steps."""
import ast
from collections import Counter
import json
import hashlib
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from elm_pipeline import tasks
from elm_pipeline import evaluation
from elm_pipeline.io import sha, write


class TaskContract(unittest.TestCase):
    def test_source_syntax(self):
        for path in (REPO/"elm_pipeline").rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
            compile(path.read_text(), str(path), "exec", dont_inherit=True)

    def test_raw_window_ids_survive_machine_moves_not_time_changes(self):
        # Compile only these stdlib metadata functions, without importing MNE,
        # Torch, data, or the preprocessing module on the developer laptop.
        source = ast.parse((REPO/"elm_pipeline/adapters/scalp_native200.py").read_text())
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("safe_id", "window_record_id")]
        context = {"re": re, "hashlib": hashlib, "WINDOW_SAMPLES": 1000}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "metadata_fixture", "exec"), context)
        key = context["window_record_id"]
        a = SimpleNamespace(subject="sub-01", session="closed", path=Path("/machine_a/raw/eeg/a.set"))
        b = SimpleNamespace(subject="sub-01", session="closed", path=Path("/machine_b/raw/eeg/a.set"))
        self.assertEqual(key("ADFTD", a, Path("/machine_a/raw"), 0),
                         key("ADFTD", b, Path("/machine_b/raw"), 0))
        self.assertNotEqual(key("ADFTD", a, Path("/machine_a/raw"), 0),
                            key("ADFTD", a, Path("/machine_a/raw"), 1000))

    def test_no_label_bin_or_answer_leak(self):
        self.assertEqual(tasks.no_leak_fixture()["status"], "passed")

    def test_bad_records_cannot_erase_a_source_or_class(self):
        parsed = ast.parse((REPO/"elm_pipeline/features.py").read_text())
        functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "retained_coverage"]
        context = {"Counter": Counter}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "coverage_fixture", "exec"), context)
        selected = [{"label": "a"}, {"label": "b"}, {"label": "b"}]
        check = context["retained_coverage"]
        self.assertEqual(check(selected, selected[:2])["accepted"], 2)
        for accepted in ([], selected[:1]):
            with self.assertRaises(ValueError):
                check(selected, accepted)

    def test_report_refuses_missing_duplicate_or_changed_artifacts(self):
        # Synthetic strings and metadata only; no model or data dependencies.
        with tempfile.TemporaryDirectory(prefix=".pipeline_fixture_", dir=REPO) as tmp:
            root = Path(tmp)
            folder = root/"out/evaluations/fixture"
            folder.mkdir(parents=True)
            feature = root/"features"; feature.mkdir()
            record = {"id": "fixture", "dataset": "fixture", "label": "left_hand",
                      "bins": {f: "high" for f in tasks.FIELDS}}
            write(feature/"val_index.json", {"records": [record]})
            checkpoint = root/"checkpoint.txt"; checkpoint.write_text("synthetic metadata fixture")
            cfg = {"output": str(root/"out"), "features": str(feature), "software_debug": True}
            frozen = {"config_sha256": "fixture", "files": {"checkpoint": {
                "path": str(checkpoint), "sha256": sha(checkpoint)}}}
            write(folder/"identity.json", frozen)
            rows = []
            for task in ["label"] + ["feature:"+f for f in tasks.FIELDS] + ["report"]:
                item = tasks.example(record, task)
                text = tasks.dumps(item["expected"])
                rows.append({"id": item["id"], "task": task, "dataset": "fixture",
                             "source": "projector", "question": item["question"],
                             "expected": item["expected"], "text": text,
                             "config_sha256": "fixture", "checkpoint_sha256": sha(checkpoint),
                             **tasks.score(text, item)})
            rawfile = folder/"raw.jsonl"
            def store(rr):
                rawfile.write_text("".join(json.dumps(r)+"\n" for r in rr))
            summary = {"status": "complete", "results": {"projector": evaluation.metrics(rows)}}
            write(folder/"summary.json", summary)
            store(rows)
            self.assertEqual(evaluation.report(cfg, "fixture")["status"], "rescored_available_artifacts")
            for rr in ([], rows[:-1], rows+rows[:1]):
                store(rr)
                with self.assertRaises(ValueError):
                    evaluation.report(cfg, "fixture")
            store(rows)
            write(folder/"summary.json", {"status": "complete", "results": {}})
            with self.assertRaises(ValueError):
                evaluation.report(cfg, "fixture")
            write(folder/"summary.json", summary)
            checkpoint.write_text("changed fixture")
            with self.assertRaises(ValueError):
                evaluation.report(cfg, "fixture")

    def test_report_format_is_separate_from_accuracy(self):
        r = {"id": "fixture", "dataset": "fixture", "label": "left_hand",
             "bins": {f: "high" for f in tasks.FIELDS}}
        item = tasks.example(r, "report")
        wrong = json.loads(tasks.dumps(item["expected"]))
        wrong["label"] = "EEG Report"
        score = tasks.score(tasks.dumps(wrong), item)
        self.assertTrue(score["valid"])
        self.assertFalse(score["correct"])
        self.assertFalse(score["components"]["label"])
        wrong["statistics"].pop(tasks.FIELDS[0])
        self.assertFalse(tasks.score(tasks.dumps(wrong), item)["valid"])


if __name__ == "__main__":
    unittest.main()
