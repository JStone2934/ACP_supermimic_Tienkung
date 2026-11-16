"""Independent physical acceptance metrics for adjustment-step standing aiming."""
import numpy as np


def segment_metrics(error, feet, dt=.02):
    hits = np.isfinite(error) & (error <= .05)
    edges = np.diff(np.r_[False, hits, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    qualifying = starts[(ends-starts-1)*dt >= 5.-1e-8]
    capture = int(qualifying[0]) if len(qualifying) else None
    captured = capture is not None and capture*dt <= 3.+1e-8
    later = np.arange(len(error))*dt >= 3.
    rate = float(hits[later].mean()) if later.any() else 0.
    drift = (float(np.linalg.norm(feet[capture:]-feet[capture], axis=-1).max())
             if captured else None)
    return dict(passed=bool(captured and rate >= .95 and drift <= .05),
                capture_s=capture*dt if capture is not None else None,
                post_grace_hit_rate=rate, stable_foot_drift_m=drift)


def episode_metrics(root, feet, tilt, error, finite_state, terminated, *, switching=False, dt=.02):
    arrays = (root, feet, tilt)
    finite = all(np.isfinite(x).all() for x in arrays) and bool(np.asarray(finite_state).all())
    full = len(error) == 1501
    radius = float(np.linalg.norm(root[:, :2]-root[0, :2], axis=-1).max())
    max_tilt = float(np.max(tilt))
    cuts = [(0,500),(500,1000),(1000,1501)] if switching else [(0,len(error))]
    segments = [segment_metrics(error[a:b], feet[a:b]) for a,b in cuts if b <= len(error)]
    standing = full and finite and not terminated and radius <= .5 and max_tilt <= 15.
    passed = standing and len(segments) == len(cuts) and all(s['passed'] for s in segments)
    return dict(passed=bool(passed), standing_pass=bool(standing), frames=len(error),
                full_30_seconds=full, terminated=bool(terminated), finite=bool(finite),
                root_activity_radius_m=radius, max_tilt_deg=max_tilt, segments=segments)
