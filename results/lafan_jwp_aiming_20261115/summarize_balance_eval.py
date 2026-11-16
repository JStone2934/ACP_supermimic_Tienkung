"""Summarize completed physical balance evaluations without changing their pass flags."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    report=json.loads(path.read_text());trajectory=np.load(path.with_suffix('.npz'))
    assert str(trajectory['checkpoint_sha256'])==report['checkpoint_sha256']
    assert report['level']==0 and not report['aim_required']
    assert report['seconds']==30 and report['continuous_without_reset'] and report['reference_free']
    feet=trajectory['feet'];tilt=trajectory['tilt'];root=trajectory['root']
    dt=float(trajectory['control_dt']);assert len(feet)==1501 and abs(dt-.02)<1e-8
    standing=np.array([row['standing_pass'] for row in report['scenarios']],dtype=bool)
    settle=round(3/dt);late=round(20/dt)
    drift=np.linalg.norm(feet[settle:]-feet[settle],axis=-1).max(axis=(0,2))
    speed=np.linalg.norm(np.diff(feet[late:],axis=0)/dt,axis=-1).mean(axis=(0,2))
    crossings={}
    for threshold in [15,30,45]:
        exceeded=tilt>threshold;ever=exceeded.any(0)
        crossings[str(threshold)]=dict(episodes=int(ever.sum()),
            median_first_crossing_s=float(np.median(exceeded[:,ever].argmax(0)*dt)) if ever.any() else None)
    return dict(report=str(path.resolve()),checkpoint_sha256=report['checkpoint_sha256'],seed=report['seed'],
        full_gate_passed=report['passed'],full_success_rate=float(np.mean([row['passed'] for row in report['scenarios']])),
        balance_only_success_rate=float(standing.mean()),median_max_tilt_deg=float(np.median(tilt.max(0))),
        median_max_root_drift_m=float(np.median(np.linalg.norm(root[...,:2]-root[0,...,:2],axis=-1).max(0))),
        median_max_foot_drift_after3s_m=float(np.median(drift)),mean_foot_speed_20_30s_m_s=float(speed.mean()),
        balanced_episodes=int(standing.sum()),
        balanced_episodes_median_foot_drift_m=float(np.median(drift[standing])) if standing.any() else None,
        balanced_episodes_mean_foot_speed_m_s=float(speed[standing].mean()) if standing.any() else None,
        tilt_crossings=crossings,
        median_tilt_at_seconds={str(t):float(np.median(tilt[round(t/dt)])) for t in [0,.16,.32,.64,1,2,3,10,20,30]},
        median_root_height_at_seconds={str(t):float(np.median(root[round(t/dt),:,2])) for t in [0,1,2,3,10,20,30]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,nargs='+',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=dict(scope='Diagnostic only. Balance-only excludes foot stability. Low foot motion after falling is not standing success.',
                rows=[summarize(p) for p in args.reports])
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for row in result['rows']:
        print(json.dumps({k:row[k] for k in ['seed','full_gate_passed','balance_only_success_rate','balanced_episodes_median_foot_drift_m','tilt_crossings']}))


if __name__=='__main__':main()
