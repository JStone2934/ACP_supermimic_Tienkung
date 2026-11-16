"""Translate learned leg coordinates and static load to a COM-preserving stance."""
import argparse,json
from pathlib import Path
import torch
from supermimic.acp.aiming_fullbody import FullbodyPolicy, validate_checkpoint
from supermimic.acp.aiming_fullbody_io import save_checkpoint, json_save
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.robots import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','geometry','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    ck=torch.load(a.source,map_location='cpu',weights_only=False)
    report=json.loads(a.geometry.read_text());assert sha256(a.source)==report['source_sha256']
    task=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,task)
    assert list(task.robot.joint_names)==report['joint_names']
    old=FullbodyPolicy(task.robot);old.load_state_dict(ck['pi'])
    assert not ck['pi']['dec.8.weight'][22:29].count_nonzero()
    torch.manual_seed(271);h,c=torch.randn(16,10,99),torch.randn(16,107)
    a.out.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(report['rows'],1):
        out=a.out/f'ckpt_{i:06d}.pt';assert not out.exists()
        assert row['maximum_solver_residual']<1e-6 and row['max_leg_q_delta_rad']<.05
        assert row['independent_fk_position_error_m']<2e-6 and row['independent_fk_rotation_matrix_error']<2e-6
        q=torch.tensor(row['initial_q']);dq=torch.tensor(row['leg_q_delta_rad'])
        torch.testing.assert_close(q[:12]-task.initial_q[:12],dq,atol=2e-8,rtol=1e-5)
        torch.testing.assert_close(q[12:],task.initial_q[12:],atol=0,rtol=0)
        assert ((q>=task.mdp.soft_lo)&(q<=task.mdp.soft_hi)).all()
        delta=torch.tensor(row['delta_leg_pd_target_rad'])/task.alpha[:12]
        pi={k:v.clone() for k,v in ck['pi'].items()}
        pi['dec.8.bias'][:12]+=delta
        for columns,shift in ((list(range(12)),dq),(list(range(64,76)),delta)):
            flat=[32+t*99+k for t in range(10) for k in columns]
            pi['dec.0.bias']-=pi['dec.0.weight'][:,flat]@(shift/old.obs_norm.std()[columns]).repeat(10)
        new=FullbodyPolicy(task.robot);new.load_state_dict(pi)
        masked=h.clone();masked[:,:,:12]-=dq;masked[:,:,64:76]-=delta
        with torch.no_grad():
            expected=old(masked,c);expected[:,:12]+=delta
            torch.testing.assert_close(new(h,c),expected,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(new(h,c)[:,22:29],old(h,c)[:,22:29],atol=0,rtol=0)
        for k,v in ck['pi'].items():
            if k not in ('dec.0.bias','dec.8.bias'):assert torch.equal(v,pi[k]),k
        cfg=dict(ck['config'],initial_q=row['initial_q'],initialization_mode='com_preserving_pose_calibration')
        calibration=dict(method=__doc__,fraction=row['fraction'],geometry_sha256=sha256(a.geometry),
            leg_q_delta_rad=row['leg_q_delta_rad'],leg_pd_delta_rad=row['delta_leg_pd_target_rad'],
            foot_shift_xy_m=row['foot_shift_xy_m'],translated_history_equivalence=True,
            right_arm_actions_exact=True,normalizers_exact=True,reset_noise_unchanged=True,
            shared_initial_pose_for_all_targets=True,physical_acceptance=False)
        result=dict(ck,pi=pi,stage='calibration',config=cfg,task_config=WideAimingTask('cpu',cfg).metadata(),
                    source_checkpoint=dict(path=str(a.source.resolve()),sha256=sha256(a.source)),
                    calibration=calibration)
        validate_checkpoint(result,WideAimingTask('cpu',cfg));save_checkpoint(out,result)
        json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=calibration))
        print(out,flush=True)


if __name__=='__main__':main()
