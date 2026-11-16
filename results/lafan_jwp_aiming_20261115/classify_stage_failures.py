"""Describe failures without excluding any scenario or changing report pass flags."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[]
for path in a.reports:
 r=json.loads(path.read_text());failed=[]
 for i,s in enumerate(r['scenarios']):
  if s['passed']:continue
  reasons=[]
  if not s['standing_pass']:reasons.append('standing')
  for seg in s['segments']:
   t=seg.get('capture_s')
   if t is None or t>3:reasons.append('capture')
   if seg.get('post_grace_hit_rate',0)<.95:reasons.append('hit_rate')
   drift=seg.get('stable_foot_drift_m')
   if drift is not None and drift>.05:reasons.append('foot_drift')
  failed.append(dict(index=i,fixed=i<r['fixed_count'],parameters=r['target_parameters'][0][i],reasons=sorted(set(reasons)),metrics=s))
 rows.append(dict(seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],full_report_passed=r['passed'],failures=failed))
a.out.write_text(json.dumps(dict(scope=__doc__,rows=rows),indent=2));print(json.dumps([dict(seed=r['seed'],failures=len(r['failures']),standing_failures=sum('standing' in x['reasons'] for x in r['failures'])) for r in rows]))
