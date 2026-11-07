"""Check stage ordering, checkpoint selection, restart, and failure handling without GPUs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import train_acp_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def exercise(self, directory, calls, fail=False):
        root = directory / "run"
        source = directory / "motions" / "test"
        source.mkdir(parents=True, exist_ok=True)
        for i in range(40):
            for suffix in ("_motion.npz", "_terrain.npy"):
                path = source / (str(i) + suffix)
                if not path.exists(): path.write_bytes(b"fixture")
        robot = SimpleNamespace(robot_id="g1", motion_dir=str(source.parent), metadata=lambda: {"robot_id":"g1","model_hash":"test"})

        def value(cmd, flag):
            return cmd[cmd.index(flag) + 1]

        def execute(cmd, **kwargs):
            module = cmd[cmd.index("-m") + 1]
            calls.append(cmd)
            if fail:
                return SimpleNamespace(returncode=1)
            if module.endswith(("train_stage1", "train_stage2")):
                out = Path(value(cmd, "--log_dir"))
                (out / "last.pt").touch()
                if out.name == "stage1_ft":
                    for n in (1000, 2000, 3000, 4000):
                        (out / f"ckpt_{n:06d}.pt").touch()
            if module.endswith("evaluate"):
                checkpoint = Path(value(cmd, "--ckpt"))
                scores = {"ckpt_001000": (20, .8), "ckpt_002000": (10, .7),
                          "ckpt_003000": (10, .8), "ckpt_004000": (30, .9), "last": (5, .8)}
                failures, reward = scores[checkpoint.stem]
                Path(value(cmd, "--out")).write_text(json.dumps({"n_failed": failures, "norm_return": reward}))
            return SimpleNamespace(returncode=0)

        with patch.object(pipeline, "get_robot", return_value=robot), patch.object(pipeline, "REPO", directory), patch.object(pipeline.os, "chdir"), \
             patch.object(pipeline.sys, "argv", ["pipeline", "--run-dir", str(root)]), \
             patch.object(pipeline.subprocess, "run", side_effect=execute), \
             patch.object(pipeline.subprocess, "Popen", return_value=SimpleNamespace(pid=123, wait=lambda: 0)), \
             contextlib.redirect_stdout(io.StringIO()):
            pipeline.main()
        return root

    def test_selection_transition_and_idempotent_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            directory = Path(tmp)
            root = self.exercise(directory, calls)
            chosen = json.loads((root / "selected_tracker.json").read_text())["selected"]
            self.assertTrue(chosen["checkpoint"].endswith("ckpt_003000.pt"))
            stage2 = [c for c in calls if "supermimic.acp.train_stage2" in c]
            self.assertEqual(len(stage2), 2)
            self.assertNotIn("--resume", stage2[0])
            self.assertIn("--resume", stage2[1])
            self.assertIn("--eta_rew_max", stage2[1])
            tracker = next(c for c in calls if "supermimic.acp.train_planner_offline" in c)
            self.assertIn(chosen["checkpoint"], tracker)
            count = len(calls)
            self.exercise(directory, calls)
            self.assertEqual(len(calls), count, "Restart must not repeat finished stages")
            self.assertEqual(json.loads((root / "status.json").read_text())["state"], "complete")

    def test_changed_data_cannot_reuse_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); calls = []
            self.exercise(directory, calls)
            (directory / "motions/test/0_motion.npz").write_bytes(b"changed")
            count = len(calls)
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                self.exercise(directory, calls)
            self.assertEqual(len(calls), count)

    def test_changed_command_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); calls = []
            root = self.exercise(directory, calls)
            marker = root / "stage1.done.json"
            data = json.loads(marker.read_text()); data["command_hash"] = "stale"
            marker.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "stale completion"):
                self.exercise(directory, calls)

    def test_failure_does_not_advance_and_restart_uses_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root = directory / "run"
            (root / "stage1").mkdir(parents=True)
            (root / "stage1" / "last.pt").touch()
            calls = []
            with self.assertRaises(RuntimeError):
                self.exercise(directory, calls, fail=True)
            self.assertEqual(len(calls), 1)
            self.assertIn("--resume", calls[0])
            self.assertFalse((root / "stage1.done.json").exists())
            self.assertFalse((root / "selected_tracker.json").exists())
            self.assertEqual(json.loads((root / "status.json").read_text())["state"], "failed")


if __name__ == "__main__":
    unittest.main()
