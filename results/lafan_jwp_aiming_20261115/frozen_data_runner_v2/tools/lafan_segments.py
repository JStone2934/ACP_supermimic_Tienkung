"""Conservatively screen repaired references, then recheck complete segments.

Use the protected tools.convert_php entry, never run this script as a batch
outside the memory queue. All source/repair clips remain intact.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation

from tools.jwp_pipeline import ROOT, POLICY, configure, sha, atomic_json, atomic_npz, runs, quality_and_labels, verify_loaders
from tools.lafan_pipeline import initialize_gpu, gpu_fk, check as check_native

VERSION="lafan-repaired-screened-v1"
SELECTION=dict(min_frames=251,edge_margin_frames=2,body_displacement_guard_m=.098,
               body_displacement_p95_max_m=.05,body_displacement_max_m=.10,
               penetration_guard_m=.002,joint_velocity_guard_ratio=.98,
               source_support_distance_guard_m=.01,source_support_slip_guard_m_s=.05)


def scan(row,parent,baseline,out,acp_root):
    configure(acp_root,ROOT/".cache/GMR")
    from supermimic.acp.robots import get_robot
    robot=get_robot("tienkung3_29dof");key=row["key"]
    with np.load(Path(parent)/"raw"/(key+".npz")) as z: support=z["source_contact"]
    labels,quality,diag=quality_and_labels(Path(row["motion"]),robot,support,diagnostics=True)
    with np.load(row["motion"]) as z: motion=z["joint_pos"];positions=z["body_pos_w"];names=z["body_names"]
    with np.load(Path(baseline)/"motions"/(key+"_motion.npz")) as z:
        np.testing.assert_array_equal(names,z["body_names"])
        distance=np.linalg.norm(positions-z["body_pos_w"],axis=-1)
    bad=(diag["terrain_penetration_m"]>SELECTION["penetration_guard_m"]) | (diag["self_penetration_m"]>SELECTION["penetration_guard_m"])
    bad |= diag["joint_velocity_ratio"].max(-1)>SELECTION["joint_velocity_guard_ratio"]
    bad |= diag["joint_limit_violation_rad"].max(-1)>POLICY["joint_limit_tolerance_rad"]
    bad |= distance.max(-1)>SELECTION["body_displacement_guard_m"]
    bad |= np.quantile(distance,.95,axis=1)>SELECTION["body_displacement_p95_max_m"]
    # Limit one-sided derivatives as well, since each cut creates new endpoints.
    edge_bad=(np.abs(np.diff(motion[:,7:],axis=0))*50/robot.velocity).max(-1)>SELECTION["joint_velocity_guard_ratio"]
    bad[:-1] |= edge_bad;bad[1:] |= edge_bad
    bad |= ((np.abs(labels["contact_dist"][0,:,:2])>SELECTION["source_support_distance_guard_m"])&support).any(-1)
    foot_speed=np.linalg.norm(np.diff(labels["body_pos_w"][:,:2,:2],axis=0)*50,axis=-1)
    sliding=((foot_speed>SELECTION["source_support_slip_guard_m_s"])&support[1:]&support[:-1]).any(-1)
    bad[1:] |= sliding;bad[:-1] |= sliding
    bad=binary_dilation(bad,iterations=SELECTION["edge_margin_frames"],border_value=1)
    intervals=[(int(a),int(b)) for a,b in runs(~bad) if b-a>=SELECTION["min_frames"] and support[a:b].any()]
    atomic_npz(Path(out)/"parent_frame_checks"/(key+".npz"),**diag,body_displacement_m=distance,excluded_frame=bad)
    report=dict(key=key,frames=len(motion),candidate_intervals=intervals,selection=SELECTION,
                parent_rejection_reasons=quality["rejection_reasons"],candidate_frames=sum(b-a for a,b in intervals))
    atomic_json(Path(out)/"screening"/(Path(key).name+".json"),report)
    return [dict(parent_entry=row,start=a,end=b) for a,b in intervals]


def prepare(job,parent,baseline,out):
    row=dict(job["parent_entry"]);parent_key=row["key"];a,b=job["start"],job["end"]
    key=f"{parent_key}__f{a:06d}_{b:06d}"
    out=Path(out)
    try:
        with np.load(row["motion"]) as z: payload=dict(z)
        motion=payload["joint_pos"][a:b].copy()
        payload.update(joint_pos=motion,**gpu_fk(motion))
        with np.load(Path(baseline)/"motions"/(parent_key+"_motion.npz")) as z:
            np.testing.assert_array_equal(z["body_names"],payload["body_names"])
            displacement=np.linalg.norm(payload["body_pos_w"]-z["body_pos_w"][a:b],axis=-1)
            reference=z["joint_pos"][a:b]
        angular=(Rotation.from_quat(reference[:,3:7],scalar_first=True).inv()*Rotation.from_quat(motion[:,3:7],scalar_first=True)).magnitude()
        fidelity=dict(body_displacement_p95_m=float(np.percentile(displacement,95)),body_displacement_max_m=float(displacement.max()),
                      root_translation_max_m=float(np.linalg.norm(motion[:,:3]-reference[:,:3],axis=-1).max()),
                      root_rotation_max_rad=float(angular.max()),waist_angle_change_max_rad=float(np.abs(motion[:,19:22]-reference[:,19:22]).max()))
        provenance=dict(row["provenance"],source_frame_interval=[a,b],parent_repaired_key=parent_key,
                        parent_repaired_motion=row["motion"],parent_repaired_motion_sha256=row["motion_sha256"],
                        segmentation="contiguous native-QC and fidelity screen, 2-frame padding, minimum 5 seconds; no retiming or blending")
        payload.update(provenance=json.dumps(provenance),data_version=VERSION,repair=json.dumps(fidelity))
        path=out/"motions"/(key+"_motion.npz");atomic_npz(path,**payload)
        terrain=path.with_name(Path(key).name+"_terrain.npy");np.save(terrain,np.zeros((0,1,10),np.float32))
        with np.load(Path(parent)/"raw"/(parent_key+".npz")) as z:
            atomic_npz(out/"raw"/(key+".npz"),qpos=z["qpos"][a:b],source_contact=z["source_contact"][a:b],source_root=z["source_root"][a:b],fps=50.)
        row.update(key=key,parent_key=parent_key,parent_interval=[a,b],frames=b-a,motion=str(path),terrain=str(terrain),
                   provenance=provenance,repair_stats=fidelity,state="prepared",
                   axis_audit=dict(row["axis_audit"],scope="Initial 31-to-29 mapping, before repair; current segment FK recomputed"))
    except Exception:
        row.update(key=key,state="failed",error=traceback.format_exc())
    return row


def check(row,out,acp_root):
    row=check_native(row,out,acp_root)
    if row["state"]!="converted":return row
    q=json.loads(Path(row["quality"]).read_text());f=row["repair_stats"]
    q.update(repair_fidelity=f,repair_fidelity_policy=dict(body_displacement_p95_max_m=.05,body_displacement_max_m=.10,
                                                        root_rotation="preserved",waist_angles="preserved"))
    if f["body_displacement_p95_m"]>.05 or f["body_displacement_max_m"]>.10 or f["root_rotation_max_rad"]>1e-5 or f["waist_angle_change_max_rad"]>1e-6:
        q["rejection_reasons"].append("repair_fidelity")
    q["strict_quality_pass"]=not q["rejection_reasons"]
    atomic_json(row["quality"],q)
    row.update(strict_quality_pass=q["strict_quality_pass"],rejection_reasons=q["rejection_reasons"],quality_sha256=sha(row["quality"]))
    atomic_json(Path(out)/"states"/(Path(row["key"]).name+".json"),row)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lafan-segment-from",type=Path,required=True)
    ap.add_argument("--baseline",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--acp-root",type=Path,default=ROOT.parent/"standing-aiming")
    ap.add_argument("--devices",default="0,1")
    ap.add_argument("--workers-per-gpu",type=int,default=1)
    ap.add_argument("--cpu-workers",type=int,default=2)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--scan",action="store_true")
    args=ap.parse_args()
    for k in ("lafan_segment_from","baseline","out","acp_root"):setattr(args,k,getattr(args,k).resolve())
    parent=args.lafan_segment_from;out=args.out
    manifest=json.loads((parent/"manifest.json").read_text())
    assert manifest["failed_count"]==0
    baseline=json.loads((args.baseline/"manifest.json").read_text())
    base_rows={r["key"]:r for r in baseline["entries"]}
    devices=["cuda:"+d for d in args.devices.split(",") for _ in range(args.workers_per_gpu)]
    if not devices or args.cpu_workers<1:ap.error("positive CPU/GPU worker counts required")
    for row in manifest["entries"]:
        for field in ("motion","terrain","labels","quality"):
            assert sha(row[field])==row[field+"_sha256"]
        base=base_rows[row["key"]]
        assert sha(base["motion"])==base["motion_sha256"]
    if args.scan:
        print(json.dumps(dict(parent_clips=len(manifest["entries"]),reserve_gib=2+9*len(devices)+6*args.cpu_workers,selection=SELECTION)));return
    identity=dict(version=VERSION,parent_manifest_sha256=sha(parent/"manifest.json"),baseline_manifest_sha256=sha(args.baseline/"manifest.json"),
                  policy=POLICY,selection=SELECTION,script_sha256=sha(__file__),
                  code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"tools").glob("*.py"))},
                  packager_sha256=sha(ROOT/"scripts/tiangong3/package_lafan_acp.py"),
                  acp_code={str(p.relative_to(args.acp_root)):sha(p) for p in sorted((args.acp_root/"supermimic/acp").glob("*.py"))})
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True,exist_ok=True)
    with (out/"controller.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"plan.json").exists():
            if not args.resume or json.loads((out/"plan.json").read_text())["identity_hash"]!=identity_hash:raise ValueError("changed identity or --resume missing")
        else:atomic_json(out/"plan.json",dict(identity=identity,identity_hash=identity_hash,command=sys.argv,workers=dict(gpu=len(devices),cpu=args.cpu_workers)))
        for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
        context=mp.get_context("spawn");queue=context.Queue()
        for d in devices:queue.put(d)
        records=[]
        with ProcessPoolExecutor(args.cpu_workers,mp_context=context) as cpu:
            candidates=[]
            futures=[cpu.submit(scan,row,str(parent),str(args.baseline),str(out),str(args.acp_root)) for row in manifest["entries"]]
            for future in as_completed(futures):candidates.extend(future.result())
            atomic_json(out/"candidates.json",candidates)
            print(f"Selected {len(candidates)} continuous candidate segments; fresh FK, labels and native checks follow",flush=True)
            pending=[]
            for job in candidates:
                name=f"{Path(job['parent_entry']['key']).name}__f{job['start']:06d}_{job['end']:06d}"
                state=out/"states"/(name+".json")
                if state.exists():
                    row=json.loads(state.read_text())
                    if row["state"]=="converted":
                        for field in ("motion","terrain","labels","quality"):
                            if sha(row[field])!=row[field+"_sha256"]:raise ValueError("completed segment was changed")
                        records.append(row)
                        continue
                pending.append(job)
            with ProcessPoolExecutor(len(devices),mp_context=context,initializer=initialize_gpu,initargs=(queue,str(args.acp_root))) as gpu:
                futures=[gpu.submit(prepare,job,str(parent),str(args.baseline),str(out)) for job in pending]
                checks=[]
                for future in as_completed(futures):
                    row=future.result()
                    if row["state"]=="prepared":checks.append(cpu.submit(check,row,str(out),str(args.acp_root)))
                    else:records.append(row)
                for future in as_completed(checks):
                    row=future.result();records.append(row)
                    print(json.dumps({k:row.get(k) for k in ("key","frames","state","strict_quality_pass","rejection_reasons","error")}),flush=True)
        records.sort(key=lambda r:r["key"])
        converted=[r for r in records if r["state"]=="converted"]
        validation=verify_loaders(out,args.acp_root,ROOT/".cache/GMR") if converted else dict(passed=False)
        if converted and (validation["clips"]!=len(converted) or validation["frames"]!=sum(r["frames"] for r in converted)):raise ValueError("segment loader coverage mismatch")
        result=dict(version=VERSION,identity_hash=identity_hash,source_count=len(candidates),converted_count=len(converted),
                    upstream_source_clip_count=len(manifest["entries"]),parent_manifest=str(parent/"manifest.json"),
                    baseline_manifest=str(args.baseline/"manifest.json"),failed_count=len(records)-len(converted),
                    total_frames=sum(r["frames"] for r in converted),strict_quality_pass_count=sum(r["strict_quality_pass"] for r in converted),
                    robot_metadata=manifest["robot_metadata"],policy=POLICY,selection=SELECTION,
                    training_loader_validation=validation,entries=records,all_terminal=False,status="packaging")
        atomic_json(out/"loader_validation.json",validation)
        atomic_json(out/"manifest.json",result)
        if converted and result["failed_count"]==0:
            # ROS also provides a top-level `scripts` package: load this local
            # packager explicitly instead of resolving that unrelated package.
            import importlib.util
            spec=importlib.util.spec_from_file_location("lafan_packager",ROOT/"scripts/tiangong3/package_lafan_acp.py")
            packager=importlib.util.module_from_spec(spec);spec.loader.exec_module(packager)
            packager.package(out)
            good=[r for r in converted if r["strict_quality_pass"]]
            if good:
                accepted=verify_loaders(out/"accepted",args.acp_root,ROOT/".cache/GMR")
                assert accepted["clips"]==len(good) and accepted["frames"]==sum(r["frames"] for r in good)
                atomic_json(out/"accepted/loader_validation.json",accepted)
                result["accepted_loader_validation"]=accepted
                summary=json.loads((out/"RESULTS.json").read_text());summary["accepted_loader_validation"]=accepted
                summary["upstream_source_clip_count"]=len(manifest["entries"])
                summary["scope"]="Screened repaired intervals only; the full 60 upstream clips remain rejected as whole clips"
                atomic_json(out/"RESULTS.json",summary)
        result.update(all_terminal=True,status="completed" if not result["failed_count"] else "completed_with_failures")
        atomic_json(out/"manifest.json",result)
        print(json.dumps({k:result[k] for k in ("status","converted_count","strict_quality_pass_count","total_frames","failed_count")}),flush=True)


if __name__=="__main__":main()
