"""Read-only, training-split-only upright double-support pose screening."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.robots import get_robot,sha256
from supermimic.rotations import quat_to_mat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--shard',type=int,required=True)
    a=p.parse_args();assert a.shard in (0,1) and not a.out.exists()
    torch.set_num_threads(2);robot=get_robot('tienkung3_29dof');kin=RobotKinematics(a.device,robot)
    manifest=json.loads(a.manifest.read_text());rows=[];counts=dict(clips=0,frames=0,quiet_upright_frames=0,stable_window_frames=0)
    torso=kin.tracked_names.index(robot.torso_body)
    feet=[kin.tracked_names.index(robot.contact_bodies[k][0]) for k in ('left_foot','right_foot')]
    for number,entry in enumerate(manifest['entries']):
        if number%2!=a.shard:continue
        path=Path(entry['motion']);labels=Path(entry['labels'])
        assert sha256(path)==entry['motion_sha256'] and sha256(labels)==entry['labels_sha256']
        with np.load(path) as z:
            assert z['joint_names'].tolist()==list(robot.joint_names)
            pose=torch.tensor(z['joint_pos'],device=a.device);fps=float(np.asarray(z['fps']).reshape(-1)[0])
        with np.load(labels) as z:
            assert z['body_names'][:2].tolist()==['left_foot','right_foot']
            contact=torch.tensor(z['contact_info'][0,:,:2,0]>.5,device=a.device).all(-1)
        n=len(pose);assert pose.shape==(n,36) and len(contact)==n
        R=quat_to_mat(pose[:,3:7]);zero=torch.zeros(n,3,device=a.device)
        with torch.no_grad():bp,br,_,_=kin.forward(pose[:,:3],R,zero,zero,pose[:,7:],torch.zeros_like(pose[:,7:]))
        tilt=lambda rot:torch.rad2deg(torch.atan2(torch.linalg.vector_norm(rot[:,:2,2],dim=-1),rot[:,2,2]))
        chest=tilt(br[:,torso]);pelvis=tilt(R)
        foot_speed=torch.linalg.vector_norm(torch.gradient(bp[:,feet],spacing=(1/fps,),dim=(0,))[0],dim=-1).amax(-1)
        root_speed=torch.linalg.vector_norm(torch.gradient(pose[:,:3],spacing=(1/fps,),dim=(0,))[0],dim=-1)
        joint_speed=torch.gradient(pose[:,7:],spacing=(1/fps,),dim=(0,))[0].abs().mean(-1)
        gate=contact&(chest<=7)&(pelvis<=10)&(foot_speed<=.12)&(root_speed<=.15)
        width=max(3,round(.5*fps));width+=1-width%2
        stable=torch.nn.functional.conv1d(gate.float()[None,None],torch.ones(1,1,width,device=a.device),padding=width//2)[0,0]>=width-.1
        score=chest/10+pelvis/15+foot_speed/.12+root_speed/.15+.2*joint_speed
        score=torch.where(stable,score,torch.full_like(score,float('inf')))
        counts['clips']+=1;counts['frames']+=n;counts['quiet_upright_frames']+=int(gate.sum());counts['stable_window_frames']+=int(stable.sum())
        # Diverse clip candidates, separated in time; final selection still requires mechanical and physical checks.
        for _ in range(2):
            index=int(score.argmin())
            if not torch.isfinite(score[index]):break
            rows.append(dict(key=entry['key'],source=entry['source'],motion=str(path),motion_sha256=entry['motion_sha256'],
                labels=str(labels),labels_sha256=entry['labels_sha256'],frame=index,fps=fps,score=float(score[index]),
                chest_tilt_deg=float(chest[index]),pelvis_tilt_deg=float(pelvis[index]),
                max_foot_speed_m_s=float(foot_speed[index]),root_speed_m_s=float(root_speed[index]),
                root_pose_wxyz=pose[index,:7].cpu().tolist(),joint_pos=pose[index,7:].cpu().tolist(),
                double_support_window_seconds=width/fps))
            score[max(0,index-round(fps)):min(n,index+round(fps)+1)]=float('inf')
    rows.sort(key=lambda r:r['score']);a.out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(scope=__doc__+' No physical acceptance.',manifest_sha256=sha256(a.manifest),device=a.device,shard=a.shard,
                counts=counts,candidates=rows,validation_data_used=False,thresholds=dict(chest_deg=7,pelvis_deg=10,feet_speed=.12,root_speed=.15,continuous_contact_seconds=.5))
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
