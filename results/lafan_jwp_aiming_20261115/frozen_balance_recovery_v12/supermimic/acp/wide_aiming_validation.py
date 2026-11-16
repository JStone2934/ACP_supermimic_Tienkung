"""Independent physical acceptance metrics for adjustment-step standing aiming."""
import numpy as np


def aim_segment_metrics(error, dt=.02, radius=.05):
    hits = np.isfinite(error) & (error <= radius)
    edges = np.diff(np.r_[False, hits, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    qualifying = starts[(ends-starts-1)*dt >= 5.-1e-8]
    capture = int(qualifying[0]) if len(qualifying) else None
    captured = capture is not None and capture*dt <= 3.+1e-8
    later = np.arange(len(error))*dt >= 3.
    rate = float(hits[later].mean()) if later.any() else 0.
    return dict(passed=bool(captured and rate >= .95),
                capture_s=capture*dt if capture is not None else None,post_grace_hit_rate=rate)


def segment_metrics(error, feet, dt=.02, radius=.05):
    result=aim_segment_metrics(error,dt,radius)
    capture=result['capture_s']
    drift=(float(np.linalg.norm(feet[int(round(capture/dt)):]-feet[int(round(capture/dt))],axis=-1).max())
           if capture is not None and capture<=3.+1e-8 else None)
    # Acquisition means sustained aim AND settled feet. If the ray is already
    # on target while the robot makes its allowed initial adjustment step, the
    # start of that early hit run must not lock the foot anchor prematurely.
    hits=np.isfinite(error)&(error<=radius)
    hold=int(round(5./dt))+1
    for index in range(min(int(round(3./dt))+1,max(0,len(error)-hold+1))):
        if not hits[index:index+hold].all():continue
        candidate=float(np.linalg.norm(feet[index:]-feet[index],axis=-1).max())
        if candidate<=.05:
            capture=index*dt;drift=candidate;break
    result['capture_s']=capture
    result.update(passed=bool(result['passed'] and capture is not None and capture<=3.+1e-8 and drift is not None and drift<=.05),stable_foot_drift_m=drift)
    return result


def episode_metrics(root, feet, tilt, error, finite_state, terminated, *, switching=False, dt=.02, radius=.05, aim_required=True):
    arrays = (root, feet, tilt)
    finite = all(np.isfinite(x).all() for x in arrays) and bool(np.asarray(finite_state).all())
    full = len(error) == 1501
    root_radius = float(np.linalg.norm(root[:, :2]-root[0, :2], axis=-1).max())
    max_tilt = float(np.max(tilt))
    cuts = [(0,500),(500,1000),(1000,1501)] if switching else [(0,len(error))]
    segments = [segment_metrics(error[a:b], feet[a:b],dt=dt,radius=radius) for a,b in cuts if b <= len(error)]
    standing = full and finite and not terminated and root_radius <= .5 and max_tilt <= 15.
    passed = standing and len(segments) == len(cuts) and all(s['passed'] for s in segments)
    if not aim_required:
        settle=int(round(3/dt))
        stable_drift=float(np.linalg.norm(feet[settle:]-feet[settle],axis=-1).max()) if full else float('inf')
        passed=standing and stable_drift<=.05
    return dict(passed=bool(passed), standing_pass=bool(standing), frames=len(error),
                full_30_seconds=full, terminated=bool(terminated), finite=bool(finite),
                root_activity_radius_m=root_radius, max_tilt_deg=max_tilt, segments=segments)
