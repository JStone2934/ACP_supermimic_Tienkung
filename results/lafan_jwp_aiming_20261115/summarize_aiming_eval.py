"""Separate aiming, balance and foot stability in completed physical evaluations."""
import argparse,json
from pathlib import Path
import numpy as np
from supermimic.acp.wide_aiming_validation import aim_segment_metrics


def summarize(path):
    r=json.loads(path.read_text())
    assert r['seconds']==30 and r['continuous_without_reset'] and r['reference_free']
    with np.load(path.with_suffix('.npz')) as z:
        assert str(z['checkpoint_sha256'])==r['checkpoint_sha256']
        error=z['error'].copy();feet=z['feet'].copy();tilt=z['tilt'].copy()
        assert len(error)==1501 and abs(float(z['control_dt'])-.02)<1e-8
    balance=np.array([s['standing_pass'] for s in r['scenarios']])
    full=np.array([s['passed'] for s in r['scenarios']])
    aim=[]
    for i in range(error.shape[1]):
        cuts=[(0,500),(500,1000),(1000,1501)] if r['online_target_switches'] and i>=r['fixed_count'] else [(0,1501)]
        aim.append(all(aim_segment_metrics(error[a:b,i],radius=r['hit_radius_m'])['passed'] for a,b in cuts))
    aim=np.array(aim)
    def quantile(values,q):
        # Inf means the ray misses the target plane, and must not be silently discarded.
        value=np.quantile(values,q,method='nearest') if values.size else np.inf
        return float(value) if np.isfinite(value) else None
    stable_error=error[150:,balance];drift=np.linalg.norm(feet[150:]-feet[150],axis=-1).max(axis=(0,2))
    return dict(report=str(path.resolve()),checkpoint_sha256=r['checkpoint_sha256'],seed=r['seed'],level=r['level'],
        full_gate_passed=r['passed'],fixed_grid_all_pass=r['fixed_grid_all_pass'],random_success_rate=r['random_success_rate'],
        episodes=len(full),balanced_episodes=int(balance.sum()),aim_only_episodes=int(aim.sum()),
        both_balanced_and_aimed_episodes=int((balance&aim).sum()),fully_passed_episodes=int(full.sum()),
        balanced_postgrace_error_m_median=quantile(stable_error,.5),balanced_postgrace_error_m_p95=quantile(stable_error,.95),
        balanced_postgrace_ray_valid_fraction=float(np.isfinite(stable_error).mean()) if stable_error.size else None,
        balanced_postgrace_hit_fraction=float((stable_error<=r['hit_radius_m']).mean()) if stable_error.size else None,
        balanced_foot_drift_after3s_m_median=quantile(drift[balance],.5),median_max_tilt_deg=float(np.median(tilt.max(0))),
        balanced_error_m_median_at_seconds={str(t):quantile(error[round(t/.02),balance],.5) for t in [0,1,2,3,5,10,20,30]})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=[summarize(x) for x in a.reports]
    report=dict(scope='Diagnostics only; authoritative full pass flags remain unchanged. Aim-only excludes feet. Null error quantiles include invalid rays or an empty subset.',rows=rows)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
