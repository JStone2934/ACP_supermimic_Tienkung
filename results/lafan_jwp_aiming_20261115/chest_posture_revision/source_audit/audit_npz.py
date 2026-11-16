import json,numpy as np
from pathlib import Path
p=Path('results/lafan_jwp_aiming_20261115/chest_posture_revision'); rows=json.load(open(p/'pose_candidates_ranked.json')); out=[]
def geom(z,f):
 b=z['body_names'].tolist(); a=z['body_pos_w'][f]; ans={}
 for side in ['l','r']:
  s,e,w=[a[b.index(f'{n}_{side}_link')] for n in ['shoulder_yaw','elbow_pitch','wrist_roll']]
  v=w-s;u=e-s;t=w-e
  ans[side]={'shoulder_wrist_horizontal_deg':float(np.degrees(np.arctan2(abs(v[2]),np.linalg.norm(v[:2])))), 'elbow_bend_deg':float(np.degrees(np.arccos(np.clip(np.dot(u,t)/np.linalg.norm(u)/np.linalg.norm(t),-1,1)))), 'shoulder_wrist_distance_m':float(np.linalg.norm(v))}
 return ans
for i,r in enumerate(rows):
 if not r['source'].endswith('.npz'):continue
 s=np.load(r['source']);t=np.load(r['motion']);pr=json.loads(str(t['provenance']));f=pr['source_frame_interval'][0]+r['frame']; names=t['joint_names'].tolist();idx=[s['joint_names'].tolist().index(n) for n in names];diff=t['joint_pos'][r['frame'],7:]-s['joint_pos'][f,idx];arm=[j for j,n in enumerate(names) if any(x in n for x in ['shoulder','elbow','wrist'])];
 out.append(dict(candidate=i+1,key=r['key'],frame50=f,time_s=f/50,source=r['source'],motion=r['motion'],source_geometry=geom(s,f),target_geometry=geom(t,r['frame']),max_arm_joint_change_deg=float(np.degrees(abs(diff[arm]).max())),arm_changes_deg={names[j]:float(np.degrees(diff[j])) for j in arm}))
(p/'source_audit/npz_audit.json').write_text(json.dumps(out,indent=2))
for i in [1,2,17]:print(json.dumps(out[i]))
print('first2seconds',sum(r['time_s']<2 for r in out),len(out))
