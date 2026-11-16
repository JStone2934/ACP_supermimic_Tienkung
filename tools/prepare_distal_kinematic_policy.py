"""Attach an audited offline target network while preserving all other joint outputs."""
import argparse,json
from pathlib import Path
import torch,numpy as np,mujoco
from supermimic.acp.aiming_fullbody import FullbodyPolicy,validate_checkpoint
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.aiming_fullbody_io import save_checkpoint,json_save
from supermimic.acp.robots import sha256
p=argparse.ArgumentParser(description=__doc__)
for k in ('source','geometry-source','head','audit','out'):p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--gains',type=float,nargs='+',default=[2.,4.]);a=p.parse_args();torch.set_num_threads(2);assert all(0<=g<=9 for g in a.gains)
ck=torch.load(a.source,map_location='cpu',weights_only=False);base=torch.load(a.geometry_source,map_location='cpu',weights_only=False);head=torch.load(a.head,map_location='cpu',weights_only=False);audit=json.loads(a.audit.read_text())
assert not ck.get('smoke') and head['source_sha256']==sha256(a.geometry_source);assert audit['head_sha256']==sha256(a.head) and audit['source_sha256']==head['source_sha256'] and audit['torch_mujoco_max_error_difference_m']<1e-4
# Support-only trained sources may differ in their first six encoder columns.
for key,value in base['pi'].items():
 if key=='enc.0.weight':assert torch.equal(value[:,6:],ck['pi'][key][:,6:])
 else:assert torch.equal(value,ck['pi'][key]),key
for key in ('initial_q','initial_height','initial_quat'):assert ck['config'][key]==base['config'][key]
T=WideAimingTask('cpu',ck['config']);validate_checkpoint(ck,T);ids=T.arm_ids.tolist()[4:];assert head['joint_ids']==ids
model=T.robot.get_spec().compile();data=mujoco.MjData(model);addresses,dof=T.robot.joint_addresses(model);data.qpos[:3]=[0,0,T.initial_height];data.qpos[3:7]=T.initial_quat;data.qpos[addresses]=T.initial_q;mujoco.mj_forward(model,data);gravity=torch.tensor(data.qfrc_bias[dof]/np.array(T.robot.kp),dtype=torch.float32)
W=torch.tensor(base['calibration']['pseudoinverse'])/T.alpha[ids,None];old=FullbodyPolicy(T.robot);old.load_state_dict(ck['pi']);torch.manual_seed(197);h=torch.randn(16,10,99);c=torch.randn(16,107);body=[i for i in range(29) if i not in ids]
a.out.mkdir(parents=True,exist_ok=True)
for number,gain in enumerate(a.gains,1):
 out=a.out/f'ckpt_{number:06d}.pt';assert not out.exists();policy=FullbodyPolicy(T.robot);policy.load_state_dict(ck['pi']);policy.enable_distal_kinematic_head()
 with torch.no_grad():
  policy.distal_kinematic_head.net.load_state_dict(head['network']);policy.distal_kinematic_head.output_scale.copy_(head['half']/T.alpha[ids]);policy.distal_kinematic_head.output_bias.copy_((head['mid']+gravity[ids]-T.mdp.default_q[ids])/T.alpha[ids]-policy.dec[-1].bias[ids]);policy.distal_command_skip.weight.zero_();policy.distal_command_skip.weight[:,:3]=gain*W;policy.distal_command_skip.weight[:,3:6]=-gain*W;policy.distal_command_skip.bias.zero_();torch.testing.assert_close(policy(h,c)[:,body],old(h,c)[:,body],atol=0,rtol=0)
 cfg=dict(ck['config'],distal_kinematic_head=True,initialization_mode='audited_offline_kinematic_head_calibration',distill_iters=0,wm_only_iters=100,critic_warmup_iters=100);task=WideAimingTask('cpu',cfg)
 calibration=dict(method=__doc__,head_sha256=sha256(a.head),audit_sha256=sha256(a.audit),geometry_source_sha256=sha256(a.geometry_source),feedback_gain=gain,gravity_pd_offset_rad=gravity[ids].tolist(),other26_outputs_exact=True,normalizers_and_reset_exact=True,online_ik=False,runtime_motion_files=False,physical_acceptance=False)
 artifact=dict(task=ck['task'],stage='calibration',smoke=False,config=cfg,task_config=task.metadata(),robot_metadata=ck['robot_metadata'],pi=policy.state_dict(),wm=ck['wm'],provenance=ck.get('provenance'),source_checkpoint=dict(path=str(a.source.resolve()),sha256=sha256(a.source)),calibration=calibration)
 validate_checkpoint(artifact,task);save_checkpoint(out,artifact);json_save(out.with_suffix('.json'),dict(checkpoint_sha256=sha256(out),calibration=calibration));print(str(out),flush=True)
