import json,torch,numpy as np
from pathlib import Path
from supermimic.acp.aiming_wide import WideAimingTask
from supermimic.acp.networks import WorldModel
from supermimic.acp.aiming_fullbody import flat_boxes
from supermimic.acp.mdp import unflatten_state,normalize_xi
from supermimic.acp.standing import freeze_normalizers
root=Path('/home/js/robot/locomotion/standing-aiming');out=root/'results/lafan_jwp_aiming_20261115/chest_posture_revision/take006_training/rewardonly_component_audit';torch.set_num_threads(2);torch.set_grad_enabled(False);dev='cuda:0';ck=torch.load(root/'logs/tienkung3_29dof/lafan_jwp_20261115/take006_upright/train/ckpt_001700.pt',map_location=dev,weights_only=False);task=WideAimingTask(dev,ck['config']);wm=WorldModel(task.mdp.scan_dim,robot=task.robot).to(dev);wm.load_state_dict(ck['wm']);freeze_normalizers(wm)
results={}
for iteration in [1700,1750]:
 path=root/f'logs/tienkung3_29dof/lafan_jwp_20261115/take006_rewardonly_recovery/train/wm_validation/{iteration:06}.npz';z=np.load(path);d={k:torch.as_tensor(z[k],device=dev) for k in z.files};n=min(8,d['s'].shape[1]);starts=[32,96];S=unflatten_state(torch.cat([d['s'][t,:n] for t in starts]));initial=S
 for j in range(32):
  a=torch.cat([d['a'][t+j,:n] for t in starts]);xi=torch.cat([d['xi'][t+j,:n] for t in starts]);S=wm.step(S,a,normalize_xi(xi),task.mdp.scan(S,flat_boxes(S)))
 truth=unflatten_state(torch.cat([d['s1'][t+31,:n] for t in starts]));valid=~torch.cat([d['done'][t:t+32,:n].any(0) for t in starts])
 def measure(S):
  bp,br,_,_=task.kin.forward(*S);R=br[:,task.torso];tilt=lambda r:torch.rad2deg(torch.atan2(torch.linalg.vector_norm(r[:,:2,2],dim=-1),r[:,2,2]));return dict(chest_deg=tilt(R),pelvis_deg=tilt(S[1]),root_z_m=S[0][:,2])
 pred=measure(S);actual=measure(truth);start=measure(initial);rows={}
 for k in pred:
  e=pred[k][valid]-actual[k][valid];rows[k]=dict(initial_mean=float(start[k][valid].mean()),predicted_mean=float(pred[k][valid].mean()),physical_mean=float(actual[k][valid].mean()),bias=float(e.mean()),mae=float(abs(e).mean()),max_absolute_error=float(abs(e).max()))
 results[str(iteration)]=dict(trajectory=str(path),fixed_wm_iteration=1700,starts=starts,sample_count=2*n,valid_windows=int(valid.sum()),horizon=32,action_protocol='Replay identical recorded executed physical actions open-loop; exact recorded state and xi, no reconstructed velocity or history.',metrics=rows)
(out/'paired_open_loop.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
