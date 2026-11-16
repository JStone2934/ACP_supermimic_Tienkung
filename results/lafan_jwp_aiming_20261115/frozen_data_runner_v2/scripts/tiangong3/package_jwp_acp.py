"""Verify conversion hashes and expose only quality-passing clips as a training subset."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.jwp_pipeline import atomic_json, sha, verify_loaders


def package(out, acp_root):
    out = out.resolve()
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["failed_count"] == 0
    assert manifest["source_count"] == manifest["converted_count"]
    accepted = out/"accepted"
    columns = ["key","frames","strict_quality_pass","rejection_reasons","joint_velocity_max_ratio",
               "terrain_penetration_max_m","self_penetration_policy_max_m","self_penetration_raw_max_m",
               "left_foot_distance_p95_m","right_foot_distance_p95_m","left_foot_slip_p95_m_s","right_foot_slip_p95_m_s"]
    rows, good, bad = [], [], []
    for entry in manifest["entries"]:
        assert sha(entry["source"]) == entry["source_sha256"]
        for field in ("motion","terrain","labels","quality"):
            assert sha(entry[field]) == entry[field+"_sha256"]
        quality = json.loads(Path(entry["quality"]).read_text())
        assert entry["strict_quality_pass"] == quality["strict_quality_pass"]
        row = {k:quality.get(k,entry.get(k)) for k in columns[:8]}
        row["rejection_reasons"] = ";".join(quality["rejection_reasons"])
        for side in ("left","right"):
            support = quality["support"][side+"_foot"]
            row[side+"_foot_distance_p95_m"] = support["distance_p95_m"]
            row[side+"_foot_slip_p95_m_s"] = support["link_slip_p95_m_s"]
        rows.append(row)
        (good if entry["strict_quality_pass"] else bad).append(entry)
        if not entry["strict_quality_pass"]:
            continue
        for field, subdir in (("motion","motions/jwp"),("terrain","motions/jwp"),("labels","contacts/jwp")):
            target = Path(entry[field])
            link = accepted/subdir/target.name
            link.parent.mkdir(parents=True,exist_ok=True)
            if link.is_symlink():
                assert link.resolve() == target
            else:
                import os
                link.symlink_to(os.path.relpath(target,link.parent))
    with (out/"quality_summary.csv").open("w") as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    validation = verify_loaders(accepted,acp_root,ROOT/".cache/GMR")
    assert validation["clips"] == len(good)
    assert validation["frames"] == sum(r["frames"] for r in good)
    atomic_json(accepted/"loader_validation.json",validation)
    atomic_json(accepted/"manifest.json",dict(parent="../manifest.json",count=len(good),
        frames=validation["frames"],policy=manifest["policy"],entries=good,training_loader_validation=validation,
        scope="Passed the declared kinematic checks; PD/trained-policy tracking not run"))
    atomic_json(out/"rejected.json",dict(count=len(bad),entries=bad,
        policy="Keep for inspection; excluded from accepted training subset; no quality thresholds relaxed"))
    summary = dict(converted_count=len(good)+len(bad),frames=manifest["total_frames"],fps=50,
        strict_quality_pass_count=len(good),strict_quality_reject_count=len(bad),
        accepted_frames=validation["frames"],all_source_and_output_hashes_verified=True,
        all_motions=str(out/"motions"),all_contacts=str(out/"contacts"),
        recommended_motion_dir=str(accepted/"motions"),recommended_contacts_dir=str(accepted/"contacts"),
        all_loader_validation=manifest["training_loader_validation"]["passed"],accepted_loader_validation=True,
        axis_audit=str(out/"axis_audit.json"),quality_table=str(out/"quality_summary.csv"),
        collision_policy="Native MuJoCo; only the original two knee_pitch/ankle_roll structural pairs excluded; raw penetration reported too",
        pd_status="not_run",learned_tracking_status="not_run")
    atomic_json(out/"RESULTS.json",summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--acp-root",type=Path,default=ROOT.parent/"standing-aiming")
    args=ap.parse_args()
    package(args.dataset,args.acp_root)
