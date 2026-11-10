"""Source geometry correction must reject phantom support and retain real touch."""
import copy,json,tempfile,unittest
from pathlib import Path
import mujoco,numpy as np
from supermimic.acp.source_contacts import validate_zero_contacts,verify_source_receipts,source_validation_required
from supermimic.acp.make_labels import build_model,geom_distance
from supermimic.acp.robots import ROOT,get_robot

class SourceContactsTests(unittest.TestCase):
    def test_future_solver_versions_keep_source_validation(self):
        for version in ('php-segment-v18','php-segment-v19','php-segment-v20','php-segment-v100'):
            self.assertTrue(source_validation_required(version))
        for version in ('php-segment-v17','php-fullbody-v3'):
            self.assertFalse(source_validation_required(version))

    def fixture(self):
        row=json.loads((ROOT/'docs/php_conversion/evidence/source_zero_contacts_fixture.json').read_text())
        args=[np.asarray(row[k],dtype=np.float64) for k in ('positions','quaternions','joints','boxes')]
        labels=[np.asarray(row[k],dtype=np.int8 if k=='contact_surface' else np.float32) for k in ('contact_dist','contact_pos_w','contact_normal_w','contact_surface')]
        return row,args,labels

    def test_real_false_zeros_recovered_without_mutating_labels(self):
        row,args,labels=self.fixture();before=[x.copy() for x in labels]
        result=validate_zero_contacts('g1',*args,*labels,first=row['first'])
        for frame,expected in row['expected_left_knee'].items():
            self.assertAlmostEqual(float(result[0][int(frame)-row['first'],2]),expected,places=6)
        for x,y in zip(labels,before):np.testing.assert_equal(x,y)
        unchanged=np.abs(before[0])>1e-10
        for new,old in zip(result[:4],before):np.testing.assert_equal(new[unchanged],old[unchanged])
        self.assertEqual(result[-1]['checked_count'],int((~unchanged).sum()))
        self.assertFalse((result[0][:,2]<=.005).any())

    def test_true_ground_touch_is_retained(self):
        r=get_robot('g1');m=build_model(np.zeros((0,8)),r);d=mujoco.MjData(m);qa,_=r.joint_addresses(m)
        q=np.asarray(r.default_q)[None];quat=np.array([[1.,0.,0.,0.]]);p=np.zeros((1,3))
        d.qpos[:3]=p[0];d.qpos[3:7]=quat[0];d.qpos[qa]=q[0];mujoco.mj_kinematics(m,d)
        geoms=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name in r.contact_bodies['left_foot'][1]]
        ground=m.geom('ground').id
        p[0,2]=-min(geom_distance(m,d,g,ground,2.,np.zeros(6),strict_reference=True) for g in geoms)
        distances=np.full((1,9),.1,np.float32);distances[0,0]=0
        points=np.zeros((1,9,3));normals=points.copy();normals[:,:,2]=1;surfaces=np.zeros((1,9),np.int8)
        out=validate_zero_contacts(r,p,quat,q,np.zeros((0,8)),distances,points,normals,surfaces)
        self.assertLess(abs(float(out[0][0,0])),1e-7)
        self.assertEqual(out[3][0,0],0)

    def test_cpu_recomputation_rejects_changed_or_missing_receipt(self):
        row,args,labels=self.fixture()
        receipts={'0':validate_zero_contacts('g1',*args,*labels)[-1]}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);motion=root/'source_motion.npz';label_path=root/'labels.npz'
            np.savez(motion,joint_pos=np.concatenate(args[:3],axis=1),joint_names=get_robot('g1').joint_names,fps=[50])
            np.save(root/'source_terrain.npy',np.asarray(row['original_terrain']))
            np.savez(label_path,**{k:v[None] for k,v in zip(('contact_dist','contact_pos_w','contact_normal_w','contact_surface'),labels)})
            record=dict(source=str(motion),labels=str(label_path),source_frames=[0,len(args[0])],unique_variants=[0],label_variants=1)
            verify_source_receipts(record,receipts)
            bad=copy.deepcopy(receipts);bad['0']['queries'][0]['validated']['distance']=0
            with self.assertRaisesRegex(ValueError,'differs'):verify_source_receipts(record,bad)
            with self.assertRaisesRegex(ValueError,'incomplete'):verify_source_receipts(record,{})

if __name__=='__main__':unittest.main()
