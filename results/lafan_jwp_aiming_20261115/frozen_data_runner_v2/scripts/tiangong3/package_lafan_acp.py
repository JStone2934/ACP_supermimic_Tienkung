"""Verify hashes and summarize a native29 LAFAN conversion without relaxing QC."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.jwp_pipeline import atomic_json, sha


def package(out):
    out = out.resolve()
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["failed_count"] == 0
    assert manifest["converted_count"] == manifest["source_count"]
    assert manifest["training_loader_validation"]["passed"]
    rows, good, rejected = [], [], []
    reasons, contacts = Counter(), Counter()
    aggregate = dict(terrain_penetration_max_m=0., self_penetration_policy_max_m=0., joint_velocity_max_ratio=0.)
    audit = dict(source_target_position_max_m=0., source_target_rotation_max_rad=0., cuda_native_position_max_m=0.)
    for entry in manifest["entries"]:
        assert sha(entry["source"]) == entry["source_sha256"]
        for field in ("motion","terrain","labels","quality"):
            assert sha(entry[field]) == entry[field+"_sha256"]
        quality = json.loads(Path(entry["quality"]).read_text())
        assert entry["strict_quality_pass"] == quality["strict_quality_pass"]
        (good if entry["strict_quality_pass"] else rejected).append(entry)
        reasons.update(quality["rejection_reasons"])
        for key in aggregate: aggregate[key] = max(aggregate[key],quality[key])
        for key in audit: audit[key] = max(audit[key],entry["axis_audit"][key])
        row = dict(key=entry["key"],frames=entry["frames"],duration_s=(entry["frames"]-1)/50,
                   strict_quality_pass=quality["strict_quality_pass"],rejection_reasons=";".join(quality["rejection_reasons"]),
                   ground_shift_m=entry["provenance"]["ground_shift_m"],
                   joint_limit_max_rad=quality["joint_limit_max_rad"],
                   **{k:quality[k] for k in aggregate})
        for side in ("left","right"):
            support=quality["support"][side+"_foot"]
            row[side+"_foot_distance_p95_m"]=support["distance_p95_m"]
            row[side+"_foot_slip_p95_m_s"]=support["link_slip_p95_m_s"]
        with np.load(entry["labels"]) as z:
            assert z["contact_info"].shape == (1,entry["frames"],9,1)
            active=z["contact_info"][...,0].astype(bool)
            for key in ("contact_pos_w","contact_normal_w"):
                assert np.isfinite(z[key][active]).all()
            assert np.isfinite(z["contact_dist"]).all()
            assert np.all(z["contact_dist"][active] <= manifest["policy"]["contact_threshold_m"]+1e-7)
            contacts.update({str(n):int(z["contact_info"][0,:,i,0].sum()) for i,n in enumerate(z["body_names"])})
        rows.append(row)
        if entry["strict_quality_pass"]:
            for field,collection in (("motion","motions"),("terrain","motions"),("labels","contacts")):
                src=Path(entry[field]); dest=out/"accepted"/collection/Path(entry["key"]).parent/src.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.is_symlink(): assert dest.resolve()==src
                else: dest.symlink_to(os.path.relpath(src,dest.parent))
    with (out/"quality_summary.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(out/"rejected.json",dict(count=len(rejected),entries=rejected,quality_thresholds_relaxed=False))
    atomic_json(out/"accepted/manifest.json",dict(count=len(good),frames=sum(e["frames"] for e in good),entries=good,
                parent="../manifest.json",policy=manifest["policy"],scope="Kinematic acceptance only; dynamics not validated"))
    summary=dict(source_count=manifest["source_count"],converted_count=len(rows),failed_count=0,
                 frames=manifest["total_frames"],fps=50.,duration_s=(manifest["total_frames"]-len(rows))/50,
                 sample_duration_s=manifest["total_frames"]/50,
                 strict_quality_pass_count=len(good),strict_quality_reject_count=len(rejected),
                 full_clips_ready_for_training=bool(good),rejection_counts=dict(reasons),quality_maxima=aggregate,
                 axis_audit_maxima=audit,contact_frames=dict(contacts),all_source_and_output_hashes_verified=True,
                 motion_dir=str(out/"motions"),contacts_dir=str(out/"contacts"),
                 recommended_motion_dir=str(out/"accepted/motions") if good else None,
                 recommended_contacts_dir=str(out/"accepted/contacts") if good else None,
                 loader_validation=manifest["training_loader_validation"],
                 acceptance_scope="Strict kinematic checks using current native meshes; no PD or learned policy rollout",
                 pd_status="not_run",learned_tracking_status="not_run")
    atomic_json(out/"RESULTS.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("loader_validation","contact_frames")},indent=2))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",required=True,type=Path)
    package(ap.parse_args().dataset)
