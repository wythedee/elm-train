#!/usr/bin/env python3
"""CPU-only tests of the real archived source's portable entry point."""
import ast
import unittest
from pathlib import Path
import reproduce


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.source = (reproduce.REPO / "reference" / reproduce.SOURCE).read_text()
        self.text = reproduce.portable_source(self.source, reproduce.SOURCE, Path("/work/repro/o80"), [2, 5])
        self.tree = ast.parse(self.text)

    def test_training_rendering_sampling_and_parser_functions_unchanged(self):
        before = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(self.source).body
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        after = {n.name: ast.dump(n, include_attributes=False) for n in self.tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name in ["train", "prepare", "data_batch", "ExpandedOracle", "ce_objective", "distillation",
                     "rollouts", "evaluate", "promotion_gate", "smoke_roundtrip", "cleanup", "spawn_identity"]:
            self.assertEqual(before[name], after[name], name)

    def test_original_teacher_wait_removed_but_complete_assets_and_locks_required(self):
        pipeline = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "pipeline")
        text = ast.unparse(pipeline)
        self.assertNotIn("bootstrap_dispatch", text)
        self.assertIn("teacher/summary.json", text)
        self.assertIn("flock", text)
        self.assertIn("first_launch.json", text)
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "stage"
                             for n in ast.walk(pipeline)), 6)

    def test_gpu_allowlist_and_original_reserve_budget_checks_preserved(self):
        self.assertNotIn("range(2, 8)", self.text)
        for name in ["gpus", "stage", "finalize"]:
            node = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertIn("(2, 5)", ast.unparse(node), name)
        stage = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage")
        text = ast.unparse(stage)
        for required in ["first_started", "12 * 3600", "4 * 3600", "8 * 1024", "below >= 3", "cleanup"]:
            self.assertIn(required, text)
        self.assertNotIn("Reference snapshot only.", self.text)


if __name__ == "__main__":
    unittest.main()
