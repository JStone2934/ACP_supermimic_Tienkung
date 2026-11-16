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
    def exercise(self, directory, calls, fail=False, extra=(), events=None):
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
            module = cmd[cmd.index('--module') + 1] if '--module' in cmd else cmd[cmd.index("-m") + 1]
            calls.append(cmd)
            if events is not None:
                events.append((module, kwargs['env']['CUDA_VISIBLE_DEVICES']))
            if fail:
                return SimpleNamespace(returncode=1)
            if module.endswith(("train_stage1", "train_stage2", "train_stage3", "train_planner_offline")):
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

        def wait_prep():
            if events is not None:
                events.append(('prep_finished', None))
            return 0

        with patch.object(pipeline, "get_robot", return_value=robot), patch.object(pipeline, "REPO", directory), patch.object(pipeline.os, "chdir"), \
             patch.object(pipeline.sys, "argv", ["pipeline", "--run-dir", str(root), *extra]), \
             patch.object(pipeline.subprocess, "run", side_effect=execute), \
             patch.object(pipeline.subprocess, "Popen", return_value=SimpleNamespace(pid=123, wait=wait_prep, poll=lambda: 0)), \
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

    def test_dual_launch_serial_prep_and_stage3(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, events = [], []
            root = self.exercise(Path(tmp), calls, extra=('--gpus', '0,1', '--through-stage', '3', '--num-envs', '512',
                '--planner-batch', '512', '--planner-sub-batches', '4', '--planner-min-entries', '128', '--reward-chunk-size', '64'), events=events)
            self.assertEqual(events[0], ('prep_finished', None))
            training = [c for c in calls if '--module' in c]
            self.assertEqual(len(training), 6)
            for c in training:
                self.assertIn('--master-addr=127.0.0.1', c)
                self.assertIn('--nproc-per-node=2', c)
                self.assertEqual(c[c.index('--num_envs') + 1], '512')
                if 'supermimic.acp.train_stage1' not in c:
                    self.assertEqual(c[c.index('--planner_batch')+1], '512')
                    self.assertEqual(c[c.index('--planner_sub_batches')+1], '4')
                    self.assertEqual(c[c.index('--planner_min_entries')+1], '128')
                if 'supermimic.acp.train_planner_offline' not in c:
                    self.assertEqual(c[c.index('--reward_chunk_size')+1], '64')
            self.assertTrue(any('supermimic.acp.evaluate_goal' in c for c in calls))
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'complete')
            for module, devices in events[1:]:
                if '.train_' in module:
                    self.assertEqual(devices, '0,1')
                else:
                    self.assertEqual(devices, '0')

    def test_dual_failure_leaves_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaises(RuntimeError):
                self.exercise(directory, [], fail=True, extra=('--gpus', '0,1'))
            self.assertFalse((directory / 'run/stage1.done.json').exists())

    def test_offline_resume_is_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); calls = []
            planner = directory / 'run/planner'
            planner.mkdir(parents=True)
            (planner / 'last.pt').touch()
            self.exercise(directory, calls)
            command = next(c for c in calls if 'supermimic.acp.train_planner_offline' in c)
            self.assertIn('--resume', command)

    def test_invalid_sizes_fail_before_preprocessing(self):
        for extra in (('--num-envs', '3'), ('--planner-batch', '10'), ('--through-stage', '3', '--num-envs', '2')):
            with self.subTest(extra=extra), patch.object(pipeline.sys, 'argv', ['pipeline', '--run-dir', 'unused', '--gpus', '0,1', *extra]), \
                 patch.object(pipeline, 'execute_pipeline') as execute, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    pipeline.main()
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
