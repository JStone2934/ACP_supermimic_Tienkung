"""Compare accepted 224-sphere distances with native geometry on recorded policy poses."""
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
from supermimic.acp.robots import get_robot
from supermimic.acp.make_labels import build_model, geom_distance
from supermimic.acp.task import BodySpheres, ContactCfg, ContactTask

torch.set_num_threads(2)
OUT = Path('results/stage_training_analysis_20261111')
robot = get_robot('tienkung3_29dof')
sp = BodySpheres(robot, 'spheres224')
ctc = ContactTask.__new__(ContactTask)
ctc.cfg = ContactCfg(contact_geometry='spheres224')
ctc.link_idx = torch.arange(9)
ctc.sph_pts,ctc.sph_rad,ctc.sph_owner,ctc.sph_bounds = sp.pts,sp.rad,sp.owner,sp.bounds
ctc.hulls={}
ctc.has_box_terrain=True
ctc.motion=SimpleNamespace(goal_boxes=None)
with np.load('results/stage2_videos_20261111/trajectories.npz') as archive:
    d={k:archive[k] for k in archive.files}
native_rows=[]
sphere_rows=[]
sphere_all_rows=[]
ids=[]
for c,name in enumerate(d['names']):
    boxes=d['boxes'][c]
    model=build_model(boxes,robot)
    state=mujoco.MjData(model)
    qadr,_=robot.joint_addresses(model)
    body_ids=[model.body(link).id for link,_ in robot.contact_bodies.values()]
    groups=[[g for g in range(model.ngeom) if model.geom_bodyid[g] in [model.body(link).id for link in links]
             and (model.geom_contype[g] or model.geom_conaffinity[g])] for _,links in robot.contact_bodies.values()]
    terrain=[model.geom('ground').id]+[model.geom(f'box{k}').id for k,b in enumerate(boxes) if b[5]>0]
    frames=np.linspace(0,int(d['alive'][:,c].sum())-1,16).astype(int)
    for t in frames:
        state.qpos[:3]=d['root_p'][t,c]
        state.qpos[3:7]=d['root_quat'][t,c]
        state.qpos[qadr]=d['q'][t,c]
        mujoco.mj_kinematics(model,state)
        witness=np.zeros(6)
        native=np.array([min(geom_distance(model,state,g,tg,.1,witness) for g in geoms for tg in terrain) for geoms in groups])
        bp=torch.tensor(state.xpos[body_ids][None],dtype=torch.float32)
        bR=torch.tensor(state.xmat[body_ids].reshape(1,9,3,3),dtype=torch.float32)
        root=torch.tensor(d['root_p'][t,c][None],dtype=torch.float32)
        sphere=ctc.body_distance((root,),torch.tensor(boxes[None],dtype=torch.float32),fk=(bp,bR))[0].numpy()
        ctc.cfg.near_boxes=len(boxes)
        sphere_all=ctc.body_distance((root,),torch.tensor(boxes[None],dtype=torch.float32),fk=(bp,bR))[0].numpy()
        ctc.cfg.near_boxes=4
        sphere_all_rows.append(sphere_all)
        native_rows.append(native);sphere_rows.append(sphere);ids.append(dict(name=str(name),frame=int(t)))
native=np.stack(native_rows);sphere=np.stack(sphere_rows)
summary={}
for b,name in enumerate(robot.contact_bodies):
    nt=native[:,b]<=.01;st=sphere[:,b]<=.01
    near=native[:,b]<.05
    summary[name]=dict(native_touch=int(nt.sum()),sphere_touch=int(st.sum()),false_touch=int((st&~nt).sum()),
                       missed_touch=int((nt&~st).sum()),near_distance_error_p95_m=float(np.quantile(np.abs(native[near,b]-sphere[near,b]),.95)) if near.any() else None)
per_group={}
for group in sorted({row['name'].split('/')[0] for row in ids}):
    mask=np.array([row['name'].startswith(group+'/') for row in ids])
    per_group[group]=dict(poses=int(mask.sum()),native_touch_by_body=(native[mask]<=.01).sum(0).tolist(),
                          sphere_touch_by_body=(sphere[mask]<=.01).sum(0).tolist())
all_boxes=np.stack(sphere_all_rows)
result=dict(poses=len(ids),checks=len(ids)*9,threshold_m=.01,per_body=summary,per_group=per_group,sample_ids=ids,
            nearest_four_vs_all_terrain_touch_disagreements=int(((sphere<=.01)!=(all_boxes<=.01)).sum()),
            limitation='192 poses from a selected 12-clip Stage2 recording, not a full geometry or gradient validation.')
(OUT/'rollout_geometry_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='sample_ids'},indent=2))
