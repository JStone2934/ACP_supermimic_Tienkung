"""Regression tests for robot identity, name order, frames, and malformed data."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import mujoco
import numpy as np
import torch
from supermimic.acp.robots import get_robot, resolve_robot
from supermimic.acp.motion import load_php_npz, MotionLib
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.terrain import load_terrain
from supermimic.acp.geometry import fixed_transform, surface_points
from supermimic.acp.task import BodySpheres, ContactTask, ContactCfg
from supermimic.acp.mdp import MDP
import shutil
from supermimic.acp.networks import Policy, WorldModel
from supermimic.acp.buffer import ReplayBuffer
from supermimic.acp.make_labels import build_model, geom_distance


class RobotContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot=get_robot('tienkung3_29dof'); cls.meta=cls.robot.metadata()
        cls.model=cls.robot.get_spec().compile()

    def write_motion(self,folder,**changes):
        q=np.arange(29,dtype=float)/100
        jp=np.tile(np.r_[0,0,self.robot.stand_height,1,0,0,0,q],(3,1))
        fields=dict(joint_pos=jp,joint_names=np.array(self.robot.joint_names),fps=[50],robot_metadata=json.dumps(self.meta))
        fields.update(changes); p=Path(folder)/'sample_motion.npz'; np.savez(p,**fields)
        return p

    def test_named_reordering(self):
        with tempfile.TemporaryDirectory() as t:
            order=np.random.default_rng(7).permutation(29)
            jp=np.tile(np.r_[0,0,1,1,0,0,0,np.arange(29)[order]/100],(3,1))
            p=self.write_motion(t,joint_pos=jp,joint_names=np.array(self.robot.joint_names)[order])
            np.testing.assert_allclose(load_php_npz(p,self.robot)[2][0],np.arange(29)/100)

    def test_wrong_robot_and_same_dimension_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=self.write_motion(t,joint_names=np.array(get_robot('g1').joint_names))
            with self.assertRaisesRegex(ValueError,'joint names'): load_php_npz(p,self.robot)
            p=self.write_motion(t)
            with self.assertRaisesRegex(ValueError,'robot_id'): load_php_npz(p,'g1')
        with self.assertRaisesRegex(ValueError,'conflicts'): resolve_robot('g1',{'robot_metadata':self.meta})
        with self.assertRaisesRegex(ValueError,'conflicts'): resolve_robot('tienkung3_29dof',{})
        self.assertEqual(resolve_robot(None,{}).robot_id,'g1')

    def test_metadata_fields_enforced(self):
        for key in ('robot_id','model_hash','config_hash','joint_order','control_dt'):
            meta=copy.deepcopy(self.meta); meta[key]='invalid'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,key): self.robot.validate_metadata(meta)
        with self.assertRaisesRegex(ValueError,'missing'): self.robot.validate_metadata(None,legacy=True)

    def test_invalid_motion_values(self):
        with tempfile.TemporaryDirectory() as t:
            for fps in ([0],[-50],[float('nan')],[50,50]):
                with self.subTest(fps=fps):
                    with self.assertRaisesRegex(ValueError,'fps'): load_php_npz(self.write_motion(t,fps=fps),self.robot)
            names=list(self.robot.joint_names); names[0]=names[1]
            with self.assertRaisesRegex(ValueError,'joint names'): load_php_npz(self.write_motion(t,joint_names=names),self.robot)
            for arr in (np.zeros((3,35)),np.zeros((1,36)),np.full((3,36),np.nan)):
                with self.assertRaises(ValueError): load_php_npz(self.write_motion(t,joint_pos=arr),self.robot)
            jp=np.zeros((3,36)); jp[:,3]=2
            with self.assertRaisesRegex(ValueError,'quaternion'): load_php_npz(self.write_motion(t,joint_pos=jp),self.robot)

    def test_terrain_contract(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'sample_motion.npz'; terrain=Path(t)/'sample_terrain.npy'
            for x in (np.zeros((0,10)),np.zeros((1,1,10)),np.full((1,1,10),np.nan)):
                np.save(terrain,x)
                with self.assertRaises(ValueError): load_terrain(str(p))
            np.save(terrain,np.zeros((0,1,10))); self.assertEqual(load_terrain(str(p)).shape,(1,0,8))

    def test_terminal_frame_is_not_a_reset_bin(self):
        with tempfile.TemporaryDirectory() as t:
            jp=np.tile(np.r_[0,0,self.robot.stand_height,1,0,0,0,self.robot.default_q],(601,1))
            p=self.write_motion(t,joint_pos=jp)
            np.save(Path(t)/'sample_terrain.npy',np.zeros((0,1,10)))
            kin=RobotKinematics('cpu',self.robot,model=self.model)
            motion=MotionLib([str(p)],kin,'cpu',t)
            self.assertEqual(motion.num_bins,12)
            self.assertTrue(bool((motion.bin_start < motion.clip_end[motion.bin_clip]).all()))
            self.assertEqual(int(motion.frame_bin[-1]),11)

    def test_modified_motion_rejects_old_labels(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t)/'synthetic'; folder.mkdir()
            src=Path(self.robot.motion_dir)/'synthetic/standing_motion.npz'
            fields=dict(np.load(src)); fields['joint_pos']=fields['joint_pos'].copy(); fields['joint_pos'][:,7]+=.01
            dst=folder/'standing_motion.npz'; np.savez(dst,**fields)
            shutil.copy(src.with_name('standing_terrain.npy'),folder/'standing_terrain.npy')
            kin=RobotKinematics('cpu',self.robot,model=self.model); motion=MotionLib([str(dst)],kin,'cpu',t)
            mdp=MDP(kin,motion,torch.tensor(self.robot.default_q))
            cdir=str(Path(self.robot.motion_dir).parents[1]/'contacts/bootstrap')
            with self.assertRaisesRegex(ValueError,'stale contact labels'):
                ContactTask(ContactCfg(contacts_dir=cdir),mdp,t,'cpu')

    def test_degenerate_native_distance_retries_without_moving_model(self):
        root=Path(__file__).resolve().parents[1]
        case=json.loads((root/'docs/tienkung3_29dof/evidence/distance_degeneracy.json').read_text())
        m=build_model(np.array([[0,0,1,1,0,.25,.3,.5]]),self.robot); d=mujoco.MjData(m)
        qa,_=self.robot.joint_addresses(m); d.qpos[qa]=self.robot.default_q; d.qpos[:7]=case['root_qpos']
        box=m.geom('box0').id; m.geom_pos[box]=case['box_center']; mujoco.mj_kinematics(m,d)
        g=next(g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name==case['body'] and m.geom_conaffinity[g])
        before=d.geom_xpos.copy(); flags=m.opt.disableflags; witness=np.zeros(6)
        dist=geom_distance(m,d,g,box,.1,witness)
        self.assertGreater(dist,.02); self.assertLess(dist,.025)
        self.assertLess(abs(dist-np.linalg.norm(witness[:3]-witness[3:])),1e-5)
        np.testing.assert_array_equal(d.geom_xpos,before); self.assertEqual(m.opt.disableflags,flags)

    def test_model_contract(self):
        m=self.model
        self.assertEqual((m.nq,m.nv,m.nu),(36,35,0))
        self.assertEqual(sum(m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE),29)
        self.assertEqual(m.body('head_yaw_link').jntnum[0],0)
        self.assertEqual(m.body('head_pitch_link').jntnum[0],0)
        self.assertGreater(m.body('head_pitch_link').mass[0],0)
        for _,links in self.robot.contact_bodies.values():
            self.assertTrue(any((m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name in links for g in range(m.ngeom)))
        self.assertTrue(np.all(m.body_inertia[m.body_mass>0]>0))
        qadr,_=self.robot.joint_addresses(m)
        self.assertEqual(len(set(qadr)),29)

    def test_fixed_chain_and_all_primitive_types(self):
        m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="frame"><geom type="sphere" size=".1"/>
        <body name="fixed1" pos="1 0 0"><body name="fixed2" pos="0 2 0"><geom type="box" size=".1 .2 .3"/>
        <geom type="cylinder" size=".1 .2"/><geom type="capsule" size=".1 .2"/>
        <body name="active"><joint/><geom type="sphere" size=".1"/></body></body></body></body></worldbody></mujoco>''')
        p,R=fixed_transform(m,m.body('frame').id,m.body('fixed2').id)
        np.testing.assert_allclose(p,[1,2,0]); np.testing.assert_allclose(R,np.eye(3))
        with self.assertRaises(ValueError): fixed_transform(m,m.body('frame').id,m.body('active').id)
        for g in range(m.ngeom): self.assertTrue(np.isfinite(surface_points(m,g)).all())

    def test_network_dimensions_are_not_hardcoded(self):
        # Exercise a different joint count, not merely two robots both having 29 joints.
        r=replace(self.robot,joint_names=self.robot.joint_names[:3])
        pi=Policy(r.obs_dim,7,hist=2,robot=r)
        self.assertEqual(pi(torch.zeros(2,2,r.obs_dim),torch.zeros(2,7)).shape,(2,3))
        wm=WorldModel(5,hidden=(16,),robot=r)
        self.assertEqual(wm.net[-1].out_features,18)
        b=ReplayBuffer(2,2,'cpu',robot=r)
        self.assertEqual(b.data['s'].shape,(2,2,19)); self.assertEqual(b.data['a'].shape,(2,2,3))


if __name__=='__main__': unittest.main()
