"""Separate body turning/balance from aiming in an unchanged full acceptance rollout."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 r=json.loads(a.report.read_text());assert r['level'] is None and r['fixed_count']==75 and r['random_count']==100
 assert r['seconds']==30 and r['continuous_without_reset'] and r['reference_free'] and r['hit_radius_m']==.05
 with np.load(a.report.with_suffix('.npz')) as z:
  assert str(z['checkpoint_sha256'])==r['checkpoint_sha256'] and len(z['root'])==1501
  q=z['root_quat'];root=z['root'];target=z['targets'];feet=z['feet'];tilt=z['tilt']
  yaw=np.arctan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2]**2+q[...,3]**2))
  desired=np.arctan2(target[...,1]-root[...,1],target[...,0]-root[...,0]);error=np.rad2deg(np.arctan2(np.sin(yaw-desired),np.cos(yaw-desired)))
  drift=np.linalg.norm(feet[150:]-feet[150],axis=-1).max(axis=(0,2))
 basic=np.array([s['standing_pass'] for s in r['scenarios']]);full=np.array([s['passed'] for s in r['scenarios']]);angles=np.round(np.rad2deg(np.asarray(r['target_parameters'])[0,:,1])).astype(int)
 rows=[]
 for angle in [-60,-30,0,30,60]:
  indices=np.flatnonzero(angles[:75]==angle);assert len(indices)==15
  balanced=indices[basic[indices]]
  finite_median=lambda x:float(np.median(x)) if x.size and np.isfinite(x).all() else None
  rows.append(dict(azimuth_deg=angle,scenarios=15,basic_balance_passed=int(basic[indices].sum()),full_aiming_passed=int(full[indices].sum()),
   balanced_median_foot_drift_m=finite_median(drift[balanced]),
   all_median_abs_heading_error_deg_at_seconds={str(t):finite_median(np.abs(error[round(t/.02),indices])) for t in (0,1,3,10,30)},
   balanced_median_abs_heading_error_deg_at_seconds={str(t):finite_median(np.abs(error[round(t/.02),balanced])) for t in (0,1,3,10,30)},
   median_max_tilt_deg=finite_median(tilt[:,indices].max(0))))
 report=dict(scope=__doc__,source_report=str(a.report.resolve()),checkpoint_sha256=r['checkpoint_sha256'],seed=r['seed'],
  full_acceptance_passed=r['passed'],fixed_basic_balance=int(basic[:75].sum()),random_basic_balance=int(basic[75:].sum()),
  fixed_rows=rows,physical_turning_certified=False,warning='Untrained wide-range diagnostic. Heading errors of fallen cases are not evidence of successful turning. Final acceptance criteria remain unchanged.')
 a.out.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
