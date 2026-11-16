"""Queued multi-GPU repair of failed JWP references, followed by native CPU checks."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import traceback
import hashlib
import gc
import time
import numpy as np

from tools.jwp_pipeline import ROOT, POLICY, atomic_json, atomic_npz, configure, sha, quality_and_labels, verify_loaders

POLISHER=None
VERSION="native29-gpu-repair-v4-windowed"


def bounded_solve(polisher, motion, contact, settings, core_frames=3072, context_frames=256):
    """Bound GPU graph memory; preserve all source frames and expose every seam.

    Core intervals are disjoint. Context is optimized but discarded; no blending,
    resampling or label changes hide boundary artifacts from native full-clip QC.
    """
    import torch
    started=time.monotonic()
    output=np.empty_like(motion);stats=[];seams=[]
    for start in range(0,len(motion),core_frames):
        end=min(start+core_frames,len(motion))
        lo=max(0,start-context_frames);hi=min(len(motion),end+context_frames)
        try:
            solved,report=polisher.solve(motion[lo:hi],contact[lo:hi],**settings)
            output[start:end]=solved[start-lo:end-lo]
            stats.append(report)
            if start:seams.append(start)
            print(json.dumps(dict(window=[lo,hi],kept=[start,end],device=polisher.device,
                                  seconds=report['seconds'])),flush=True)
        finally:
            gc.collect()
            with torch.cuda.device(polisher.device):torch.cuda.empty_cache()
    result=dict(device=polisher.device,gpu_name=stats[0]['gpu_name'],cuda_graph=True,
                seconds=time.monotonic()-started,windows=stats,seam_frames=seams,
                core_frames=core_frames,context_frames=context_frames,
                window_boundary_policy='Unblended core extraction; fresh global derivatives and native QC',
                fidelity_p95_aggregation='Conservative maximum of window P95; segment screening recomputes exact values')
    for key in ('peak_cuda_allocated_bytes','body_displacement_p95_m','body_displacement_max_m',
                'root_translation_max_m','root_rotation_max_rad'):
        result[key]=max(s[key] for s in stats)
    return output,result


def initialize_gpu(devices,acp_root,gmr_root):
    global POLISHER
    configure(acp_root,gmr_root)
    from tools.jwp_gpu_polish import GpuPolisher
    device=devices.get()
    POLISHER=GpuPolisher(device)
    print(f"GPU repair worker ready: {device}, pid={os.getpid()}",flush=True)


def solve(entry,parent,out,settings):
    out,parent=Path(out),Path(parent)
    key=entry["key"]
    try:
        with np.load(entry["motion"]) as z:
            payload=dict(z)
        contact=np.load(parent/"raw"/(key+".npz"))["source_contact"]
        motion,stats=bounded_solve(POLISHER,payload["joint_pos"],contact,settings)
        payload.update(joint_pos=motion.astype(np.float32),data_version=VERSION,
                       parent_motion_hash=sha(entry["motion"]),repair=json.dumps(stats))
        if "body_names" in payload:
            # LAFAN exports carry full-tree states; never retain stale poses or
            # velocities from the pre-repair payload after changing qpos.
            from tools.lafan_pipeline import gpu_fk
            payload.update(gpu_fk(motion,kin=POLISHER.kin,device=POLISHER.device))
        path=out/"motions"/(key+"_motion.npz")
        atomic_npz(path,**payload)
        terrain=path.with_name(Path(entry["terrain"]).name)
        shutil.copy2(entry["terrain"],terrain)
        raw=out/"raw"/(key+".npz");raw.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(parent/"raw"/(key+".npz"),raw)
        row=dict(entry,motion=str(path),terrain=str(terrain),state="solved",parent_motion=entry["motion"],
                 parent_motion_sha256=entry["motion_sha256"],repair_stats=stats)
    except Exception:
        row=dict(entry,state="failed",error=traceback.format_exc())
    atomic_json(out/"solve_states"/(Path(key).name+".json"),row)
    return row


def check(row,out,acp_root,gmr_root):
    configure(acp_root,gmr_root)
    from supermimic.acp.robots import get_robot
    out=Path(out);key=row["key"]
    try:
        contact=np.load(out/"raw"/(key+".npz"))["source_contact"]
        labels,quality=quality_and_labels(Path(row["motion"]),get_robot("tienkung3_29dof"),contact)
        quality["repair_fidelity"]={k:row["repair_stats"][k] for k in ("body_displacement_p95_m","body_displacement_max_m","root_translation_max_m","root_rotation_max_rad")}
        quality["repair_fidelity_policy"]=dict(body_displacement_p95_max_m=.05,body_displacement_max_m=.10,root_rotation="preserved")
        if row["repair_stats"]["body_displacement_p95_m"]>.05 or row["repair_stats"]["body_displacement_max_m"]>.10:
            quality["rejection_reasons"].append("repair_fidelity")
            quality["strict_quality_pass"]=False
        labels_path=out/"contacts"/(key+".npz")
        quality_path=out/"quality"/(key+".json")
        atomic_npz(labels_path,source_motion=key+".npz",**labels)
        atomic_json(quality_path,quality)
        row.update(labels=str(labels_path),quality=str(quality_path),state="converted",
                   strict_quality_pass=quality["strict_quality_pass"],rejection_reasons=quality["rejection_reasons"])
        for field in ("motion","terrain","labels","quality"):row[field+"_sha256"]=sha(row[field])
    except Exception:row.update(state="failed",error=traceback.format_exc())
    atomic_json(out/"states"/(Path(key).name+".json"),row)
    return row


def reuse(entry,parent,out):
    row=dict(entry,reused=True)
    key=entry["key"]
    for field,destination in (("motion",out/"motions"/(key+"_motion.npz")),
                              ("terrain",out/"motions"/Path(key).parent/Path(entry["terrain"]).name),
                              ("labels",out/"contacts"/(key+".npz")),
                              ("quality",out/"quality"/(key+".json"))):
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(entry[field],destination)
        row[field]=str(destination)
    raw=out/"raw"/(key+".npz");raw.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(parent/"raw"/(key+".npz"),raw)
    atomic_json(out/"states"/(Path(key).name+".json"),row)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repair-from",type=Path,required=True)
    ap.add_argument("--reuse-repairs",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--acp-root",type=Path,default=ROOT.parent/"standing-aiming")
    ap.add_argument("--gmr-root",type=Path,default=ROOT/".cache/GMR")
    ap.add_argument("--devices",default="0,1")
    ap.add_argument("--workers-per-gpu",type=int,default=1)
    ap.add_argument("--cpu-workers",type=int,default=2)
    ap.add_argument("--iterations",type=int,default=800)
    ap.add_argument("--collision-rounds",type=int,default=8)
    ap.add_argument("--collision-iterations",type=int,default=400)
    ap.add_argument("--collision-weight",type=float,default=200000.)
    ap.add_argument("--only",nargs="+")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--scan",action="store_true")
    args=ap.parse_args()
    for name in ("repair_from","out","acp_root","gmr_root"):setattr(args,name,getattr(args,name).resolve())
    if args.cpu_workers<1 or args.workers_per_gpu<1:ap.error("positive CPU/GPU worker counts required")
    parent=args.repair_from;out=args.out
    manifest=json.loads((parent/"manifest.json").read_text())
    entries=manifest["entries"]
    if args.only:
        entries=[r for r in entries if Path(r["key"]).name in args.only]
        if len(entries)!=len(set(args.only)):raise ValueError("missing requested clips")
    settings=dict(iterations=args.iterations,rounds=args.collision_rounds,collision_iterations=args.collision_iterations,collision_weight=args.collision_weight)
    devices=[d if d.startswith("cuda:") else "cuda:"+d for d in args.devices.split(",")]
    for row in entries:
        for field in ("motion","terrain","labels","quality"):
            if sha(row[field])!=row[field+"_sha256"]:raise ValueError("parent output hash mismatch")
    identity=dict(version=VERSION,parent_manifest_sha256=sha(parent/"manifest.json"),keys=[r["key"] for r in entries],
                  settings=settings,policy=POLICY,acp_root=str(args.acp_root),
                  source_contacts={r["key"]:sha(parent/"raw"/(r["key"]+".npz")) for r in entries},
                  code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"tools").glob("*.py"))},
                  acp_code={str(p.relative_to(args.acp_root)):sha(p) for p in sorted((args.acp_root/"supermimic/acp").glob("*.py"))})
    existing_repairs={}
    if args.reuse_repairs:
        args.reuse_repairs=args.reuse_repairs.resolve()
        previous=json.loads((args.reuse_repairs/"plan.json").read_text())["identity"]
        if (previous["settings"]!=settings or previous["parent_manifest_sha256"]!=identity["parent_manifest_sha256"] or
            previous["code"]["tools/jwp_gpu_polish.py"]!=identity["code"]["tools/jwp_gpu_polish.py"] or
            previous["acp_code"]!=identity["acp_code"]):raise ValueError("reusable repair settings/code/parent mismatch")
        existing_repairs={r["key"]:r for r in json.loads((args.reuse_repairs/"manifest.json").read_text())["entries"]
                          if r["state"]=="converted" and r["strict_quality_pass"]}
        for r in existing_repairs.values():
            for field in ("motion","terrain","labels","quality"):
                if sha(r[field])!=r[field+"_sha256"]:raise ValueError("reusable repair hash mismatch")
        identity["reused_repair_manifest_sha256"]=sha(args.reuse_repairs/"manifest.json")
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True,exist_ok=True)
    with (out/"controller.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=out/"plan.json"
        if plan.exists():
            if not args.resume or json.loads(plan.read_text())["identity_hash"]!=identity_hash:raise ValueError("changed identity; choose new output directory")
        else:
            atomic_json(plan,dict(identity=identity,identity_hash=identity_hash,command=sys.argv,
                workers=dict(devices=devices,gpu_per_device=args.workers_per_gpu,cpu=args.cpu_workers),source_count=len(entries)))
        if args.scan:
            print(json.dumps(dict(clips=len(entries),repair_count=sum(not r["strict_quality_pass"] and r["key"] not in existing_repairs for r in entries),
                  reusable_repairs=len(existing_repairs),
                  reserve_gib=2+9*len(devices)*args.workers_per_gpu+6*args.cpu_workers)),flush=True);return
        records=[];pending=[]
        for entry in entries:
            state=out/"states"/(Path(entry["key"]).name+".json")
            if state.exists():
                row=json.loads(state.read_text())
                if row["state"]=="converted":
                    for field in ("motion","terrain","labels","quality"):
                        if sha(row[field])!=row[field+"_sha256"]:raise ValueError("saved output hash mismatch")
                    records.append(row);continue
            if entry["strict_quality_pass"]:records.append(reuse(entry,parent,out))
            elif entry["key"] in existing_repairs:records.append(reuse(existing_repairs[entry["key"]],args.reuse_repairs,out))
            else:pending.append(entry)
        for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[name]="1"
        ctx=mp.get_context("spawn");device_queue=ctx.Queue()
        for d in devices:
            for _ in range(args.workers_per_gpu):device_queue.put(d)
        with ProcessPoolExecutor(len(devices)*args.workers_per_gpu,mp_context=ctx,initializer=initialize_gpu,
                initargs=(device_queue,str(args.acp_root),str(args.gmr_root))) as gpu, ProcessPoolExecutor(args.cpu_workers,mp_context=ctx) as cpu:
            futures={gpu.submit(solve,r,str(parent),str(out),settings):"gpu" for r in sorted(pending,key=lambda x:-x["frames"])}
            while futures:
                done,_=wait(futures,return_when=FIRST_COMPLETED)
                for f in done:
                    kind=futures.pop(f);row=f.result()
                    if kind=="gpu" and row["state"]=="solved":
                        print(json.dumps(dict(key=row["key"],gpu_solved=True,stats=row["repair_stats"])),flush=True)
                        futures[cpu.submit(check,row,str(out),str(args.acp_root),str(args.gmr_root))]="cpu"
                    else:
                        records.append(row)
                        print(json.dumps({k:row.get(k) for k in ("key","state","strict_quality_pass","rejection_reasons","error")}),flush=True)
        records.sort(key=lambda r:r["key"])
        converted=[r for r in records if r["state"]=="converted"]
        failed=[r for r in records if r["state"]!="converted"]
        validation=verify_loaders(out,args.acp_root,args.gmr_root) if converted else dict(passed=False)
        if converted and (validation["clips"]!=len(converted) or validation["frames"]!=sum(r["frames"] for r in converted)):
            raise ValueError("repaired training loader changed clip/frame coverage")
        result=dict(version=VERSION,identity_hash=identity_hash,parent_manifest=str(parent/"manifest.json"),
            source_count=len(entries),converted_count=len(converted),failed_count=len(failed),total_frames=sum(r["frames"] for r in converted),
            strict_quality_pass_count=sum(r["strict_quality_pass"] for r in converted),all_terminal=True,
            status="completed_with_failures" if failed else "completed",training_loader_validation=validation,
            robot_metadata=manifest["robot_metadata"],policy=POLICY,entries=records)
        atomic_json(out/"loader_validation.json",validation)
        atomic_json(out/"manifest.json",result)
        for name in ("axis_audit.json",):
            if (parent/name).exists():shutil.copy2(parent/name,out/name)
        print(json.dumps({k:result[k] for k in ("converted_count","failed_count","strict_quality_pass_count","total_frames")}),flush=True)


if __name__=="__main__":main()
