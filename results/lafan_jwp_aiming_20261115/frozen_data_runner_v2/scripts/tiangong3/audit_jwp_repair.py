"""Check cumulative repair fidelity, unchanged accepted inputs and label identities."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.jwp_pipeline import configure,sha,atomic_json


@torch.no_grad()
def audit(original,final,acp_root):
    configure(acp_root,ROOT/".cache/GMR")
    from supermimic.acp.robots import get_robot
    from supermimic.acp.kinematics import RobotKinematics
    from supermimic.rotations import quat_to_mat
    robot=get_robot("tienkung3_29dof");kin=RobotKinematics("cuda:0",robot)
    original_manifest=json.loads((original/"manifest.json").read_text())
    final_manifest=json.loads((final/"manifest.json").read_text())
    old={r["key"]:r for r in original_manifest["entries"]}
    assert set(old)=={r["key"] for r in final_manifest["entries"]}
    rows=[]
    def points(a):
        x=torch.tensor(a,device="cuda:0",dtype=torch.float32);zero=torch.zeros_like(x[:,:3])
        return kin.forward(x[:,:3],quat_to_mat(x[:,3:7]),zero,zero,x[:,7:],torch.zeros_like(x[:,7:]),all_bodies=True)[0]
    for entry in final_manifest["entries"]:
        baseline=old[entry["key"]]
        with np.load(baseline["motion"]) as z:a=z["joint_pos"]
        with np.load(entry["motion"]) as z:
            b=z["joint_pos"]
            robot.validate_metadata(json.loads(str(z["robot_metadata"].item())))
            assert tuple(z["joint_names"])==robot.joint_names
            assert float(z["fps"][0])==50
            assert np.isfinite(b).all() and b.shape==a.shape and b.shape[1]==36
            assert np.max(np.abs(np.linalg.norm(b[:,3:7],axis=1)-1))<1e-5
        assert entry["source_sha256"]==baseline["source_sha256"]
        assert entry["provenance"]==baseline["provenance"]
        if baseline["strict_quality_pass"]:
            for field in ("motion","terrain","labels","quality"):
                assert sha(entry[field])==baseline[field+"_sha256"]
        with np.load(entry["labels"]) as z:
            assert str(z["motion_hash"].item())==sha(entry["motion"])
            assert str(z["terrain_hash"].item())==sha(entry["terrain"])
            assert z["contact_info"].shape==(1,len(b),9,1)
            mask=z["contact_info"][...,0].astype(bool)
            assert np.isfinite(z["body_pos_w"]).all()
            assert np.isfinite(z["contact_pos_w"][mask]).all()
            assert np.isfinite(z["contact_normal_w"][mask]).all()
        error=(points(a)-points(b)).norm(dim=-1)
        p95=torch.quantile(error,.95).item();peak=error.max().item()
        angle=(Rotation.from_quat(a[:,3:7],scalar_first=True).inv()*Rotation.from_quat(b[:,3:7],scalar_first=True)).magnitude().max()
        waist=float(np.max(np.abs(a[:,19:22]-b[:,19:22])))
        assert p95<=.05 and peak<=.10 and angle<1e-5 and waist<1e-6
        assert entry["strict_quality_pass"]
        rows.append(dict(key=entry["key"],frames=len(b),original_pass=baseline["strict_quality_pass"],final_pass=True,
                         cumulative_body_displacement_p95_m=p95,cumulative_body_displacement_max_m=peak,
                         root_rotation_max_rad=float(angle),waist_joint_change_max_rad=waist))
    report=dict(passed=True,clips=len(rows),frames=sum(r["frames"] for r in rows),
                originally_passing_unchanged=sum(r["original_pass"] for r in rows),
                repaired_to_pass=sum(not r["original_pass"] for r in rows),
                maximum_cumulative_body_displacement_p95_m=max(r["cumulative_body_displacement_p95_m"] for r in rows),
                maximum_cumulative_body_displacement_m=max(r["cumulative_body_displacement_max_m"] for r in rows),
                all_root_and_waist_attitudes_preserved=True,all_label_hashes_and_contact_values_valid=True,entries=rows)
    atomic_json(final/"repair_audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="entries"},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original",type=Path,required=True);ap.add_argument("--final",type=Path,required=True)
    ap.add_argument("--acp-root",type=Path,default=ROOT.parent/"standing-aiming")
    args=ap.parse_args();audit(args.original,args.final,args.acp_root)
