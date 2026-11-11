"""Bounded checkpoint swap evaluations; no model or training parameter changes."""
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
S1 = ROOT / 'logs/tienkung3_29dof/stage1_php_all_spheres224_20261110/train'
S2 = ROOT / 'logs/tienkung3_29dof/stage2_php_all_spheres224_20261110'
CASES = [
    ('s1_ref', S1/'ckpt_060000.pt', None),
    ('s1_offline', S1/'ckpt_060000.pt', S2/'offline/planner_020000.pt'),
    ('s1_joint40', S1/'ckpt_060000.pt', S2/'joint/ckpt_040000.pt'),
    ('s2_ref', S2/'joint/ckpt_040000.pt', None),
    ('s2_offline', S2/'joint/ckpt_040000.pt', S2/'offline/planner_020000.pt'),
    ('s2_joint40', S2/'joint/ckpt_040000.pt', S2/'joint/ckpt_040000.pt'),
    ('s2_joint24', S2/'joint/ckpt_024000.pt', S2/'joint/ckpt_024000.pt'),
    ('s2_joint12', S2/'joint/ckpt_012000.pt', S2/'joint/ckpt_012000.pt'),
    ('s2_joint40_repeat', S2/'joint/ckpt_040000.pt', S2/'joint/ckpt_040000.pt'),
    ('s2_joint12_half', S2/'joint/ckpt_012000.pt', S2/'joint/ckpt_012000.pt'),
]
results = []
for tag, tracker, planner in CASES:
    output = OUT / (tag + '.json')
    cmd = [sys.executable, '-m', 'supermimic.acp.evaluate', '--ckpt', str(tracker),
           '--motion_dir', str(OUT/'eval_motions'), '--contacts_dir',
           str(ROOT/'data/tienkung3_29dof/stage1/php_all_variants_20261110/contacts'),
           '--term_window', '0', '--seed', '0', '--out', str(output)]
    if planner is not None:
        cmd += ['--planner', str(planner)]
    if tag == 's2_joint12_half':
        cmd += ['--eta_obs', '0.5']
    (OUT/(tag+'.command.json')).write_text(json.dumps(cmd, indent=2))
    if not output.exists():
        print('Starting', tag, flush=True)
        with (OUT/(tag+'.stdout.log')).open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(output.read_text())
    row = dict(case=tag, norm_return=result['norm_return'], n_failed=result['n_failed'],
               per_folder=result['per_folder'])
    print(row, flush=True)
    results.append(row)
    (OUT/'swap_summary.json').write_text(json.dumps(results, indent=2))
