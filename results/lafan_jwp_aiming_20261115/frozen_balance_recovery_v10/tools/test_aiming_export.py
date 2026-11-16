import json
import tempfile
import unittest
from pathlib import Path
from supermimic.acp.robots import sha256
from tools.export_aiming_policy import verify_acceptance,verify_video


class ExportGateTests(unittest.TestCase):
    def test_moving_requires_full_speed_and_strict_aiming_not_intermediate_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'test.pt';checkpoint.write_bytes(b'identity only')
            r=dict(checkpoint_sha256=sha256(checkpoint),scope='Full moving aiming acceptance',level=None,passed=True,
                fixed_grid_all_pass=True,fixed_count=25,random_count=100,random_success_rate=.95,seconds=30,
                continuous_without_reset=True,reference_free=True,online_target_switches=True,hit_radius_m=.05,
                maximum_speed_m_s=.3,velocity_rmse_limit_m_s=.08)
            verify_acceptance(checkpoint,[r|dict(seed=1),r|dict(seed=2)],True)
            for key,value in [('maximum_speed_m_s',.1),('velocity_rmse_limit_m_s',.1),('level',3),('hit_radius_m',.1),('reference_free',False)]:
                with self.assertRaises(ValueError):verify_acceptance(checkpoint,[r|dict(seed=1),r|dict(seed=2)|{key:value}],True)

    def test_video_must_match_passed_scenario_and_immutable_rollout(self):
        with tempfile.TemporaryDirectory() as folder:
            b=Path(folder);video=b/'test.mp4';video.write_bytes(b'fixture identity')
            report=b/'evaluation.json';trajectory=report.with_suffix('.npz');trajectory.write_bytes(b'trajectory fixture')
            digest='a'*64
            meta=dict(checkpoint_sha256=digest,seconds=30.,hit_radius_m=.05,video_sha256=sha256(video),
                trajectory=str(trajectory),trajectory_sha256=sha256(trajectory),scenario=0)
            video.with_suffix('.json').write_text(json.dumps(meta))
            verify_video(video,[report],[dict(scenarios=[dict(passed=True)])],digest)
            with self.assertRaises(ValueError):verify_video(video,[report],[dict(scenarios=[dict(passed=False)])],digest)
            trajectory.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_video(video,[report],[dict(scenarios=[dict(passed=True)])],digest)


if __name__=='__main__':unittest.main()
