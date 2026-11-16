"""Whole-trajectory CUDA refinement; native MuJoCo remains the collision oracle."""
from __future__ import annotations
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import mujoco
import torch

from tools.jwp_pipeline import BASELINE_PAIRS, runs


class GpuPolisher:
    def __init__(self, device):
        from supermimic.acp.robots import get_robot
        from supermimic.acp.kinematics import RobotKinematics
        self.device = device
        torch.cuda.set_device(device)
        self.robot = get_robot("tienkung3_29dof")
        self.kin = RobotKinematics(device, self.robot)
        spec = self.robot.get_spec()
        for g in spec.geoms:
            if g.conaffinity: g.contype = 1
        spec.worldbody.add_geom(name="polish_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,1])
        self.model, self.data = spec.compile(), None
        self.data = mujoco.MjData(self.model)
        self.qa, _ = self.robot.joint_addresses(self.model)
        self.feet = [self.kin.body_names.index(f"ankle_roll_{s}_link") for s in ("l","r")]
        self.foot_indices = torch.tensor(self.feet, device=device)
        # Convex hull vertices preserve the exact support function of each mesh.
        self.clouds = []
        self.owners = []
        for g in range(self.model.ngeom):
            body = self.model.body(int(self.model.geom_bodyid[g])).name
            if not self.model.geom_conaffinity[g] or body not in self.kin.body_names:
                continue
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("native TienKung collision meshes required")
            mid = self.model.geom_dataid[g]
            a,n = self.model.mesh_vertadr[mid],self.model.mesh_vertnum[mid]
            points = self.model.mesh_vert[a:a+n].astype(np.float64)
            points = points[ConvexHull(points).vertices]
            points = points@Rotation.from_quat(self.model.geom_quat[g],scalar_first=True).as_matrix().T+self.model.geom_pos[g]
            self.clouds.append(self.tensor(points))
            self.owners.append(self.kin.body_names.index(body))
        self.foot_clouds = [self.clouds[self.owners.index(b)] for b in self.feet]

    def tensor(self, a):
        return torch.as_tensor(np.asarray(a),dtype=torch.float32,device=self.device)

    def fk(self,p,R,q):
        zero=torch.zeros_like(p)
        return self.kin.forward(p,R,zero,zero,q,torch.zeros_like(q),all_bodies=True)[:2]

    def witnesses(self,motion):
        rows=[]
        for t,row in enumerate(motion):
            self.data.qpos[:7],self.data.qpos[self.qa]=row[:7],row[7:]
            mujoco.mj_forward(self.model,self.data)
            for c in self.data.contact:
                bids=[int(self.model.geom_bodyid[g]) for g in c.geom]
                names=[self.model.body(b).name for b in bids]
                # Match the project's native collision refinement trigger. The
                # checker independently accepts <=2 mm persistent penetration;
                # trying to separate sub-threshold articulation contacts can
                # deform the source posture without improving acceptance.
                if frozenset(names) in BASELINE_PAIRS or c.dist>=-.0015:
                    continue
                normal=c.frame[:3].copy()
                points=[c.pos-normal*c.dist/2,c.pos+normal*c.dist/2]
                owners=[self.kin.body_names.index(n) if b else -1 for n,b in zip(names,bids)]
                local=[self.data.xmat[b].reshape(3,3).T@(p-self.data.xpos[b]) if b else p for p,b in zip(points,bids)]
                rows.append((t,owners,local,normal))
        if not rows: return None
        ts,owners,local,normals=zip(*rows)
        return (torch.tensor(ts,device=self.device),torch.tensor(owners,device=self.device),self.tensor(local),self.tensor(normals))

    def solve(self,motion,contact,iterations=800,rounds=8,collision_iterations=400,collision_weight=200000.):
        from supermimic.rotations import quat_to_mat,mat_to_quat,so3_exp
        started=time.monotonic()
        torch.cuda.reset_peak_memory_stats(self.device)
        source=self.tensor(motion)
        p0,R0,q0=source[:,:3],quat_to_mat(source[:,3:7]),source[:,7:]
        with torch.no_grad(): P0,B0=self.fk(p0,R0,q0)
        targets=P0[:,self.feet].detach().cpu().numpy()
        rotations=B0[:,self.feet].detach().cpu().numpy()
        for b in range(2):
            for a,z in runs(contact[:,b]):
                targets[a:z,b,:2]=np.median(targets[a:z,b,:2],axis=0)
                yaw=Rotation.from_matrix(rotations[a:z,b]).mean().as_euler("xyz")[2]
                rotations[a:z,b]=Rotation.from_euler("z",yaw).as_matrix()
                points=self.foot_clouds[b].cpu().numpy()
                targets[a:z,b,2]=.001-np.min(points@rotations[a,b,2])
        target=self.tensor(targets); target_R=self.tensor(rotations)
        contacts=self.tensor(contact); adjacent=self.tensor(contact[1:]&contact[:-1])
        dp=torch.zeros_like(p0,requires_grad=True)
        dr=torch.zeros_like(p0,requires_grad=True)
        q=q0.detach().clone().requires_grad_()
        velocity=self.tensor(self.robot.velocity)
        lr=torch.tensor(.006,device=self.device)
        metrics=torch.zeros(5,device=self.device)
        witnesses=None
        all_points=torch.cat(self.clouds)
        all_owners=torch.tensor([b for b,cloud in zip(self.owners,self.clouds) for _ in range(len(cloud))],device=self.device)
        # The feet use exact mesh extrema; all collision meshes also receive floor clearance gradients.
        def make_step():
            opt=torch.optim.Adam((dp,dr,q),lr=lr,capturable=True)
            def step():
                opt.zero_grad(set_to_none=True)
                R=R0@so3_exp(dr)
                P,B=self.fk(p0+dp,R,q)
                feet=P[:,self.foot_indices]
                support=(((feet-target).square().sum(-1)*contacts).sum()/contacts.sum().clamp(min=1))
                foot_ori=(((B[:,self.foot_indices]-target_R).square().sum((-1,-2))*contacts).sum()/contacts.sum().clamp(min=1))
                # Only Z is needed for the flat-plane support function.
                height=P[:,all_owners,2]+(B[:,all_owners,2,:]*all_points).sum(-1)
                floor=torch.relu(.0007-height).square().amax(-1).mean()
                slip_v=(feet[1:,:,:2]-feet[:-1,:,:2])*50
                slip=(torch.relu(slip_v.square().sum(-1).clamp(min=1e-12).sqrt()-.015).square()*adjacent).sum()/adjacent.sum().clamp(min=1)
                speed=torch.relu((q[1:]-q[:-1]).abs()*50-.85*velocity).square().mean()
                track=(P-P0).square().mean()+.01*(B-B0).square().mean()
                smooth=(q[2:]-2*q[1:-1]+q[:-2]).square().mean()
                root_smooth=((dp[1:]-dp[:-1])*50).square().mean()+.01*((dp[2:]-2*dp[1:-1]+dp[:-2])*2500).square().mean()
                loss=30000*support+1000*foot_ori+100000*floor+300*slip+10*speed+1000*track+3*smooth+.1*root_smooth
                loss=loss+2*(q-q0).square().mean()+100*dr.square().mean()+dp.square().mean()
                collision=loss*0
                if witnesses is not None:
                    ts,owners,local,normals=witnesses
                    points=P[ts[:,None],owners.clamp(min=0)]+(B[ts[:,None],owners.clamp(min=0)]@local[...,None]).squeeze(-1)
                    points=torch.where((owners<0)[...,None],local,points)
                    gap=((points[:,1]-points[:,0])*normals).sum(-1)
                    violations=torch.relu(.0002-gap).square()
                    peaks=torch.zeros(group_count,device=self.device).scatter_reduce(0,groups,violations,reduce="amax")
                    collision=peaks.sum()/len(q)
                    loss=loss+collision_weight*collision
                loss.backward();opt.step()
                with torch.no_grad():
                    q.clamp_(self.kin.q_lo+.0001,self.kin.q_hi-.0001)
                    # Preserve the calibrated root/torso attitude exactly.
                    dr.zero_()
                    q[:,12:15].copy_(q0[:,12:15])
                    metrics.copy_(torch.stack((loss,support,floor,slip,collision)).detach())
            return step
        def optimize(n,rate):
            lr.fill_(rate)
            step=make_step()
            stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):step()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):step()
            for i in range(n-3):
                lr.fill_(.00002+(rate-.00002)*(1+np.cos(np.pi*i/max(n-4,1)))/2)
                graph.replay()
            if not bool(torch.isfinite(metrics).all()):raise ValueError("nonfinite GPU refinement")
        def result():
            with torch.no_grad():return torch.cat((p0+dp,mat_to_quat(R0@so3_exp(dr)),q),-1).cpu().numpy()
        optimize(iterations,.006)
        history=[]
        for index in range(rounds):
            found=self.witnesses(result())
            if found is None:break
            witnesses=found if witnesses is None else tuple(torch.cat((old,new)) for old,new in zip(witnesses,found))
            group_keys,groups=torch.unique(torch.cat((witnesses[0][:,None],torch.sort(witnesses[1],dim=-1).values),dim=1),dim=0,return_inverse=True)
            group_count=len(group_keys)
            optimize(collision_iterations,.0015)
            history.append(dict(round=index,witnesses=len(found[0]),loss=metrics[0].item()))
        output=result()
        with torch.no_grad():
            P,B=self.fk(p0+dp,R0@so3_exp(dr),q)
            fidelity=(P-P0).norm(dim=-1)
        stats=dict(device=self.device,gpu_name=torch.cuda.get_device_name(self.device),cuda_graph=True,
                   iterations=iterations,collision_iterations=collision_iterations,collision_rounds=history,
                   collision_weight=collision_weight,
                   seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                   body_displacement_p95_m=torch.quantile(fidelity,.95).item(),body_displacement_max_m=fidelity.max().item(),
                   root_translation_max_m=dp.norm(dim=-1).max().item(),root_rotation_max_rad=dr.norm(dim=-1).max().item(),
                   final_objectives=metrics.cpu().tolist())
        return output,stats
