"""Root continuity must reject jumps even when geometry/arms/torso could pass."""
import copy, unittest
import numpy as np
from supermimic.acp.root_continuity import POLICY, score_root_positions
from tools.audit_php_variant_coverage import coverage
from tools import test_php_fidelity as fidelity_tests

class RootContinuityTests(unittest.TestCase):
    def source(self):
        t=np.arange(200)/50
        return np.stack((t,.2*np.sin(t),.8+.03*np.sin(2*t)),-1)

    def test_constant_morphology_offset_and_common_rotation_are_invariant(self):
        source=self.source();target=source+np.array([.1,-.2,.3])
        self.assertTrue(score_root_positions(source,target)['passed'])
        target[:,0]+=.02*np.sin(np.arange(len(source))/30)
        turn=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        a=score_root_positions(source,target);b=score_root_positions(source@turn.T+3,target@turn.T+3)
        for key in a['metrics']:self.assertAlmostEqual(a['metrics'][key],b['metrics'][key],places=8)

    def test_single_frame_teleport_and_frozen_root_rejected(self):
        source=self.source();target=source.copy();target[100,0]+=.13
        result=score_root_positions(source,target)
        self.assertFalse(result['passed'])
        self.assertIn('added_velocity_max_m_s',result['rejection_reasons'])
        self.assertIn('added_acceleration_max_m_s2',result['rejection_reasons'])
        self.assertFalse(score_root_positions(source,np.repeat(source[:1],len(source),axis=0))['passed'])

    def test_all_frames_and_original_sampling_are_required(self):
        source=self.source()
        for target,fps in ((source[:-1],50),(source,25),(source*np.nan,50)):
            with self.assertRaises(ValueError):score_root_positions(source,target,fps)
        target=source.copy();target[-1,2]+=.2
        self.assertFalse(score_root_positions(source,target)['passed'])

    def test_root_failure_cannot_be_omitted_from_quality_conjunction(self):
        from supermimic.acp.root_continuity import verify_report, required
        from supermimic.acp.retarget_fidelity import verify_fidelity_report
        source=self.source();target=source.copy();target[100,0]+=.13
        measured=score_root_positions(source,target)
        row=dict(geometry_pass=True,root_continuity_pass=False,quality_pass=False)
        report=dict(fidelity_policy=None,posture_policy=None,root_continuity_policy=POLICY,root_continuity=measured,variants=[row])
        verify_fidelity_report(report,None)
        row['quality_pass']=True
        with self.assertRaises(ValueError):verify_fidelity_report(report,None)
        bad=copy.deepcopy(measured);bad['passed']=True;bad['rejection_reasons']=[]
        with self.assertRaises(ValueError):verify_report(bad)
        self.assertTrue(required('php-segment-v21'));self.assertTrue(required('php-segment-v100'))
        self.assertFalse(required('php-segment-v20'))

    def test_strict_coverage_rejects_missing_failed_or_misbound_root_evidence(self):
        index,inventory=fidelity_tests.CoverageTests().fixture()
        with self.assertRaisesRegex(ValueError,'root continuity'):coverage(index,inventory,require_root_continuity=True)
        index['root_continuity_policy']=POLICY;index['root_continuity_reports']={}
        for e in index['entries']:
            e.update(motion_sha256='m',root_continuity_key='s:m')
        report=score_root_positions(self.source(),self.source())
        report.update(source_sha256='s',motion_sha256='m')
        index['root_continuity_reports']['s:m']=report
        self.assertTrue(coverage(index,inventory,require_root_continuity=True)['all_original_variants_pass'])
        for field,value in (('passed',False),('motion_sha256','other'),('source_sha256','other')):
            bad=copy.deepcopy(index);bad['root_continuity_reports']['s:m'][field]=value
            with self.assertRaises(ValueError):coverage(bad,inventory,require_root_continuity=True)

if __name__=='__main__':unittest.main()
