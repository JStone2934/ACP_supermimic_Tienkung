"""Physical 30-second moving acceptance with world-fixed target switches."""
import numpy as np
from .wide_aiming_validation import aim_segment_metrics


def moving_episode_metrics(root,velocity,command,path_goal,tilt,error,finite_state,terminated,dt=.02,maximum_speed=.3):
    arrays=(root,velocity,command,path_goal,tilt)
    finite=all(np.isfinite(x).all() for x in arrays) and bool(np.asarray(finite_state).all())
    full=len(error)==1501
    rmse=float(np.sqrt(np.mean(np.sum((velocity-command)**2,axis=-1))))
    path_error=float(np.linalg.norm(root[:,:2]-path_goal[:,:2],axis=-1).max())
    max_tilt=float(np.max(tilt))
    cuts=[(0,500),(500,1000),(1000,1501)]
    segments=[aim_segment_metrics(error[a:b],dt=dt,radius=.05) for a,b in cuts if b<=len(error)]
    commanded_distance=float(np.linalg.norm(command[:-1,:2],axis=-1).sum()*dt)
    # The protocol must actually command movement and a stop, so standing still
    # cannot pass a mislabeled all-zero moving evaluation.
    speeds=np.linalg.norm(command[:,:2],axis=-1)
    movement_tested=bool((speeds>=maximum_speed-.001).any() and (speeds<=1e-6).any()
                        and speeds.max()<=maximum_speed+.000001 and np.abs(command[:,2]).max()<=1e-6
                        and commanded_distance>=2.*maximum_speed/.3)
    physical=full and finite and not terminated and max_tilt<=15. and rmse<=.08 and path_error<=.5
    return dict(passed=bool(physical and movement_tested and len(segments)==3 and all(x['passed'] for x in segments)),
                full_30_seconds=full,finite=bool(finite),terminated=bool(terminated),max_tilt_deg=max_tilt,
                velocity_rmse_m_s=rmse,path_error_max_m=path_error,commanded_distance_m=commanded_distance,
                movement_and_stop_tested=movement_tested,segments=segments,
                feet_criterion='Stepping is required; stationary global foot anchors do not apply')
