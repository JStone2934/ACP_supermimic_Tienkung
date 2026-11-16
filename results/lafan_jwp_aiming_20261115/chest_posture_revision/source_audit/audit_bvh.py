import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/js/robot/locomotion/SuperMimic_TienKung/.cache/GMR/general_motion_retargeting/utils')
from lafan_vendor.extract import read_bvh
from lafan_vendor.utils import quat_fk
p=Path('results/lafan_jwp_aiming_20261115/chest_posture_revision/source_audit'); out=[]
for name,fr in [('aiming2_subject2',12),('walk1_subject1',45),('aiming1_subject4',13)]:
 if not (p/f'{name}.bvh').exists():continue
 d=read_bvh(str(p/f'{name}.bvh'));q,pos=quat_fk(d.quats,d.pos,d.parents);g={}
 for side in ['Left','Right']:
  s,e,w=[pos[:,d.bones.index(side+n)] for n in ['Arm','ForeArm','Hand']];v=w-s;u=e-s;t=w-e
  horizontal=np.degrees(np.arctan2(abs(v[:,1]),np.linalg.norm(v[:,[0,2]],axis=1)));bend=np.degrees(np.arccos(np.clip(np.sum(u*t,axis=1)/np.linalg.norm(u,axis=1)/np.linalg.norm(t,axis=1),-1,1)));g[side]=(horizontal,bend)
 mask=np.logical_and.reduce([g[s][0]<20 for s in g]+[g[s][1]<40 for s in g]);changes=np.where(~mask)[0]
 r=dict(thresholds=dict(horizontal_deg=20,elbow_bend_deg=40),name=name,frames=len(pos),frame30=fr,time_s=fr/30,bones=d.bones,selected={s:dict(horizontal_deg=float(g[s][0][fr]),elbow_bend_deg=float(g[s][1][fr])) for s in g},initial_tpose_like_end_frame=int(changes[0]) if len(changes) else len(pos),initial_tpose_like_duration_s=float(changes[0]/30) if len(changes) else len(pos)/30)
 out.append(r);print(json.dumps(r))
(p/'bvh_audit.json').write_text(json.dumps(out,indent=2))
