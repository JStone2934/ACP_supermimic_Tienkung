exec(open('/tmp/audit_bvh.py').read().split('for name,fr')[0])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
rows=json.load(open(p.parent/'pose_candidates_ranked.json'));fig,axes=plt.subplots(3,3,figsize=(12,10))
for ii,(ci,name,fr) in enumerate([(2,'aiming2_subject2',12),(3,'walk1_subject1',45),(18,'aiming1_subject4',13)]):
 r=rows[ci-1];t=np.load(r['motion']);s=np.load(r['source']);pr=json.loads(str(t['provenance']));sf=pr['source_frame_interval'][0]+r['frame'];d=read_bvh(str(p/f'{name}.bvh'));_,bp=quat_fk(d.quats,d.pos,d.parents)
 for col in range(3):
  ax=axes[ii,col];ax.set_aspect('equal');ax.set_xlim(-.9,.9);ax.set_ylim(-.6,.7);ax.grid(alpha=.2);ax.set_title(f'{ci:02} '+['Original BVH','31-joint source','29-joint repaired'][col]);
  if col==0:
   root=bp[fr,d.bones.index('Spine2')]; lateral=bp[fr,d.bones.index('LeftArm')]-bp[fr,d.bones.index('RightArm')];lateral[1]=0;lateral/=np.linalg.norm(lateral);delta=bp[fr]-root;pose=np.stack([delta@lateral,delta[:,1]],axis=-1)
   for side in ['Left','Right']:
    v=pose[[d.bones.index(side+n) for n in ['Shoulder','Arm','ForeArm','Hand']]]/100;ax.plot(v[:,0],v[:,1],'o-')
  else:
   z,f=(s,sf) if col==1 else (t,r['frame']);bn=z['body_names'].tolist();a=z['body_pos_w'][f];q=z['body_quat_w'][f,bn.index('pelvis')];R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();a=(a-a[bn.index('waist_pitch_link')])@R
   for side in ['l','r']:
    v=a[[bn.index(f'{n}_{side}_link') for n in ['shoulder_pitch','shoulder_yaw','elbow_pitch','wrist_roll']]];ax.plot(v[:,1],v[:,2],'o-')
  ax.set_xlabel('body lateral (m)');ax.set_ylabel('height relative chest (m)')
fig.suptitle('Arm geometry audit: source motion already starts with arms spread\nBVH displayed nearest 30 Hz sample; NPZ sampled at exact candidate 50 Hz frame');fig.tight_layout();fig.savefig(p/'three_layer_comparison.png',dpi=150)
