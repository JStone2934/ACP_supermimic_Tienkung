"""Small, source-audited G1 PHP -> TienKung task-space trajectory retargeting.

The pilot deliberately uses only terrain variant zero and exports no G1 caches.
Independent quality gates and free-root dynamics are in check_php_pilot.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import binary_closing
import torch
from supermimic.acp.robots import get_robot, sha256
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import load_php_npz
from supermimic.acp.geometry import surface_points, fixed_transform, quat_mat
from supermimic.acp.terrain import load_terrain, box_sdf
from supermimic.rotations import quat_to_mat, mat_to_quat, so3_exp

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/php_parkour'
VERSION='php-pilot-taskspace-v2'


def runs(mask):
    e=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))


def selections():
    def loco(prefix): return next((SOURCE/'motions-locomotion').glob(prefix+'*_motion.npz'))
    return [('standing','stand',loco('cfg0000'),0,100),
            ('walking','forward_a',loco('cfg0002'),100,240),
            ('walking','forward_b',loco('cfg0014'),100,160),
            ('turning','left45',loco('cfg0000'),100,170),
            ('turning','right45',loco('cfg0003'),100,200),
            ('low_step','forward_a',SOURCE/'motions-low-step/forward_walk0.5s_step0s_motion.npz',0,None),
            ('low_step','forward_b',SOURCE/'motions-low-step/forward_walk0.8s_step0s_motion.npz',0,None)]


def link_cloud(robot, name):
    m=robot.get_spec().compile(); frame,links=robot.contact_bodies[name]; fid=m.body(frame).id; points=[]
    for link in links:
        bid=m.body(link).id; p,R=fixed_transform(m,fid,bid)
        for g in range(m.ngeom):
            if m.geom_bodyid[g]==bid and m.geom_conaffinity[g]:
                v=surface_points(m,g,spacing=.015)
                points.append(p+(m.geom_pos[g]+v@quat_mat(m.geom_quat[g]).T)@R.T)
    return np.concatenate(points)


def sole(robot,name):
    pts=link_cloud(robot,name)
    # Center of the near-planar lowest sole region, not the ankle origin.
    low=pts[pts[:,2]<pts[:,2].min()+.006]
    return np.r_[(low[:,:2].min(0)+low[:,:2].max(0))/2,pts[:,2].min()]


def support_height(xy,boxes):
    h=np.zeros(len(xy))
    for cx,cy,top,c,s,hx,hy,hz in boxes:
        d=xy-[cx,cy]; local=d@np.array([[c,-s],[s,c]])
        inside=(np.abs(local[:,0])<=hx)&(np.abs(local[:,1])<=hy)
        h=np.where(inside,np.maximum(h,top),h)
    return h


class Retargeter:
    def __init__(self,device='cuda:0'):
        self.dev=device; self.src=get_robot('g1'); self.dst=get_robot('tienkung3_29dof')
        self.sk=RobotKinematics(device,self.src); self.tk=RobotKinematics(device,self.dst)
        self.sf=[self.sk.tracked_names.index(self.src.contact_bodies[n][0]) for n in ('left_foot','right_foot')]
        self.tf=[self.tk.tracked_names.index(self.dst.contact_bodies[n][0]) for n in ('left_foot','right_foot')]
        self.sc=self.tensor(np.array([sole(self.src,n) for n in ('left_foot','right_foot')]))
        self.tc=self.tensor(np.array([sole(self.dst,n) for n in ('left_foot','right_foot')]))
        # Reduced clouds only guide optimization; full MuJoCo collision is an independent gate.
        self.cloud=[]
        for n in ('left_foot','right_foot'):
            x=link_cloud(self.dst,n); idx=[int(x[:,2].argmin())]
            dist=np.linalg.norm(x-x[idx[0]],axis=1)
            for _ in range(95):
                j=int(dist.argmax());idx.append(j);dist=np.minimum(dist,np.linalg.norm(x-x[j],axis=1))
            self.cloud.append(self.tensor(x[idx]))
        z=self.tensor(np.zeros((1,3)));I=torch.eye(3,device=device)[None]
        self.sn=self.fk(self.sk,z,I,self.tensor(self.src.default_q)[None])
        self.tn=self.fk(self.tk,z,I,self.tensor(self.dst.default_q)[None])

    def tensor(self,x): return torch.tensor(x,dtype=torch.float32,device=self.dev)

    @staticmethod
    def fk(kin,p,R,q):
        return kin.forward(p,R,torch.zeros_like(p),torch.zeros_like(p),q,torch.zeros_like(q))[:2]

    def solve(self,path,start,end,iters,arm_mode="neutral"):
        pos,quat,q,fps=load_php_npz(path,self.src); assert fps==50
        end=len(pos) if end is None else end
        pos,quat,q=pos[start:end],quat[start:end],q[start:end]
        p0,R0,sq=self.tensor(pos),quat_to_mat(self.tensor(quat)),self.tensor(q)
        boxes=load_terrain(str(path))[0];boxes=boxes[boxes[:,5]>0]
        P,Rs=self.fk(self.sk,p0,R0,sq)
        foot=P[:,self.sf]+(Rs[:,self.sf]@self.sc[:,:,None]).squeeze(-1)
        foot=foot.cpu().numpy(); rotations=Rs[:,self.sf].cpu().numpy()
        src_label=ROOT/'data/php_parkour_contacts'/path.relative_to(SOURCE).with_name(path.name.replace('_motion.npz','.npz'))
        lab=np.load(src_label);dist=lab['contact_dist'][0,start:end,:2]
        contact=np.zeros_like(dist,dtype=bool)
        for b in range(2):
            speed=np.linalg.norm(np.gradient(foot[:,b],axis=0),axis=1)*fps
            contact[:,b]=(dist[:,b]<.009)|((dist[:,b]<.035)&(speed<.35))
            # Keep boundary contact states; close short internal gaps only.
            mask=contact[:,b].copy(); closed=binary_closing(mask,iterations=2);closed[:2]=mask[:2];closed[-2:]=mask[-2:]
            for a,z in runs(closed):
                if z-a<4:closed[a:z]=False
            contact[:,b]=closed
        target=foot.copy()
        for b in range(2):
            target[:,b,2]=np.maximum(target[:,b,2],support_height(target[:,b,:2],boxes))
            for a,z in runs(contact[:,b]):
                anchor=np.median(target[a:z,b],axis=0)
                anchor[2]=support_height(anchor[None,:2],boxes)[0]
                target[a:z,b]=anchor
                yaw=np.arctan2(rotations[a:z,b,1,0],rotations[a:z,b,0,0]).mean()
                c,s=np.cos(yaw),np.sin(yaw);rotations[a:z,b]=[[c,-s,0],[s,c,0],[0,0,1]]
        # Semantic target offsets account for default geometry and limb size.
        rel=(R0[:,None].transpose(-1,-2)@(P-p0[:,None]).unsqueeze(-1)).squeeze(-1)
        local=self.tn[0]+1.25*(rel-self.sn[0])
        height_delta=float(self.dst.stand_height-.789)
        proot=p0.clone();proot[:,2]+=height_delta
        body_target=proot[:,None]+(R0[:,None]@local.unsqueeze(-1)).squeeze(-1)
        orient=Rs@self.sn[1].transpose(-1,-2)@self.tn[1]
        target=self.tensor(target);contact_t=self.tensor(contact);rotations=self.tensor(rotations)
        dp=torch.zeros_like(p0,requires_grad=True);dr=torch.zeros_like(p0,requires_grad=True)
        qn=self.tensor(self.dst.default_q)[None].repeat(len(pos),1).requires_grad_()
        opt=torch.optim.Adam((dp,dr,qn),lr=.012)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iters,eta_min=.0001)
        tb=self.tensor(boxes);start_time=time.time();last={}
        for it in range(iters):
            R=R0@so3_exp(dr);pn=proot+dp
            pp,rr=self.fk(self.tk,pn,R,qn)
            fc=pp[:,self.tf]+(rr[:,self.tf]@self.tc[:,:,None]).squeeze(-1)
            foot_loss=(((fc-target)**2).sum(-1)*(1+9*contact_t)).mean()
            foot_rot=((rr[:,self.tf]-rotations)**2).mean()
            clear=0.;ground_touch=0.
            for b in range(2):
                x=pp[:,self.tf[b],None]+self.cloud[b]@rr[:,self.tf[b]].transpose(-1,-2)
                distances=x[...,2]
                for box in tb:
                    distances=torch.minimum(distances,box_sdf(x,box[None].expand(len(x),-1)))
                clear=clear+torch.relu(-distances).amax(-1).square().mean()
                mind=distances.min(-1).values
                ground_touch=ground_touch+(mind.square()*contact_t[:,b]).mean()
            body=(pp-body_target).square().mean()
            rotation=(rr[:,[7,10,13]]-orient[:,[7,10,13]]).square().mean()
            smooth=(qn[1:]-qn[:-1]).square().mean()+20*(qn[2:]-2*qn[1:-1]+qn[:-2]).square().mean()
            smooth=smooth+10*(dp[1:]-dp[:-1]).square().mean()+20*(dp[2:]-2*dp[1:-1]+dp[:-2]).square().mean()
            velocity=torch.relu((qn[1:]-qn[:-1]).abs()*50-.8*self.tensor(self.dst.velocity)).square().mean()
            loss=100*foot_loss+2*foot_rot+2*body+.05*rotation+3000*clear+100*ground_touch+.5*smooth+.5*velocity+.01*dp.square().mean()+.01*dr.square().mean()+.0001*(qn-self.tensor(self.dst.default_q)).square().mean()
            opt.zero_grad();loss.backward();opt.step();sched.step()
            with torch.no_grad():
                qn.clamp_(self.tk.q_lo+.001,self.tk.q_hi-.001)
                if arm_mode in ("neutral","clearance"):
                    qn[:,15:]=self.tensor(self.dst.default_q)[15:]
                    if arm_mode=="clearance":
                        qn[:,self.dst.joint_names.index("shoulder_roll_l_joint")]=.5
                        qn[:,self.dst.joint_names.index("shoulder_roll_r_joint")]=-.5
            if it%200==0 or it==iters-1:
                last=dict(iteration=it+1,loss=float(loss.detach()),foot_rmse=float((fc-target).square().mean().sqrt().detach()),seconds=time.time()-start_time)
                print(path.name[:40],json.dumps(last),flush=True)
        with torch.no_grad():
            pn=proot+dp;R=R0@so3_exp(dr)
        motion=np.concatenate((pn.detach().cpu().numpy(),mat_to_quat(R).cpu().numpy(),qn.detach().cpu().numpy()),axis=1)
        return motion,contact,target.cpu().numpy(),last,end,src_label


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',default='data/tienkung3_29dof/motions/php_pilot_v2');ap.add_argument('--iterations',type=int,default=1000);ap.add_argument('--device',default='cuda:0');ap.add_argument('--only',nargs='+');ap.add_argument('--arm-mode',choices=['auto','neutral','tracked','clearance'],default='auto');args=ap.parse_args()
    torch.set_num_threads(8);torch.manual_seed(0);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('output already contains a run; choose a new --out to preserve provenance')
    rt=Retargeter(args.device);meta=rt.dst.metadata();records=[]
    for category,name,path,start,end in selections():
        key=category+'/'+name
        if args.only and key not in args.only:continue
        arm_mode=("clearance" if category=="low_step" else "neutral") if args.arm_mode=="auto" else args.arm_mode
        motion,contact,target,stats,end,labels=rt.solve(path,start,end,args.iterations,arm_mode)
        folder=out/category;folder.mkdir(exist_ok=True);dest=folder/(name+'_motion.npz')
        np.savez_compressed(dest,joint_pos=motion,joint_names=np.array(rt.dst.joint_names),fps=[50],robot_metadata=json.dumps(meta),data_version=VERSION,source=str(path.relative_to(ROOT)),source_frames=[start,end])
        terrain=np.load(str(path).replace('_motion.npz','_terrain.npy'))[:,0:1]
        np.save(folder/(name+'_terrain.npy'),terrain)
        np.savez_compressed(folder/(name+'_intent.npz'),contact=contact,foot_target=target)
        src=np.load(path);cmd=src['vel_cmd'][start:end]
        record=dict(name=key,solver_version=VERSION,iterations=args.iterations,arm_mode=arm_mode,source=str(path.relative_to(ROOT)),source_hash=sha256(path),source_labels_hash=sha256(labels),source_frames=[start,end],source_command_counts={str(int(k)):int(v) for k,v in zip(*np.unique(cmd.argmax(-1),return_counts=True))},source_terrain_hash=sha256(str(path).replace('_motion.npz','_terrain.npy')),terrain_variant=0,source_family='php_atomic_stand' if category=='standing' else 'php_atomic_forward' if category=='walking' else 'php_atomic_'+name if category=='turning' else 'php_atomic_low_step',frames=len(motion),motion_hash=sha256(dest),stats=stats)
        records.append(record)
        (out/'manifest.json').write_text(json.dumps(dict(version=VERSION,robot_metadata=meta,seed=0,arm_mode=args.arm_mode,iterations=args.iterations,split='development only; insufficient independent source families',clips=records),indent=2)+'\n')
    print('wrote',out,flush=True)

if __name__=='__main__':main()
