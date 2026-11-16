"""Reload best/final standing policies and export comparison curves and best video."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--out', default='results/analysis/standing_aiming_finetune', type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, checkpoint in [('best', 'best.pt'), ('final', 'last.pt')]:
        subprocess.run([sys.executable, '-m', 'tools.evaluate_standing', '--ckpt', str(args.run / checkpoint),
                        '--out', str(args.out / name)], check=True)
    best = json.loads((args.out / 'best.json').read_text())
    final = json.loads((args.out / 'final.json').read_text())
    baseline_path = Path('results/analysis/standing_checkpoint_probe/aiming_noise0_seed0.json')
    baseline = json.loads(baseline_path.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, (label, report) in enumerate([('Source policy', baseline), ('Best fine-tuned policy', best)]):
        m = report['metrics']
        times = np.arange(len(m['root_drift_m'])) * .02
        axes[0, col].plot(times, m['root_drift_m'], label='Pelvis')
        axes[0, col].axhline(.10, color='r', linestyle='--', label='0.10 m limit')
        feet = np.asarray(m['foot_drift_m'])
        axes[1, col].plot(times, feet[:, 0], label='Left foot')
        axes[1, col].plot(times, feet[:, 1], label='Right foot')
        axes[1, col].axhline(.05, color='r', linestyle='--', label='0.05 m limit')
        axes[0, col].set_title(label)
        axes[1, col].set_xlabel('Time (s)')
    for ax in axes.flat:
        ax.set_ylabel('Horizontal displacement (m)')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Held-arm standing: continuous rollout, no reset (panels use independent y-scales)')
    fig.tight_layout()
    fig.savefig(args.out / 'drift_comparison.png', dpi=160)
    plt.close(fig)
    compact = lambda r: {k: r[k] for k in ('iteration', 'elapsed_sim_seconds', 'terminated', 'max_root_drift_m',
        'max_foot_drift_m', 'max_tilt_deg', 'held_arm_p95_error_deg', 'standing_30s_pass')}
    summary = dict(source_checkpoint=str(baseline.get('checkpoint')), baseline=compact(baseline),
                   best=compact(best), final=compact(final), run=str(args.run),
                   root_drift_reduction_fraction=1 - best['max_root_drift_m'] / baseline['max_root_drift_m'])
    (args.out / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    title = f"Held arm | iter {best['iteration']} | root {best['max_root_drift_m']:.3f} m | {'PASS' if best['standing_30s_pass'] else 'NOT PASSED'}"
    subprocess.run([sys.executable, '-m', 'supermimic.acp.render', str(args.out / 'best.npz'), str(args.out / 'best.mp4'),
                    '--seconds', '30', '--tail', '0', '--hide-reference', '--title', title], check=True)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
