"""Explicit target curricula; final acceptance remains a separate fixed protocol."""
import itertools
import math
import torch

STAGES=[
    dict(name='balance',distance=[3.,3.],azimuth=0.,elevation=0.,radius=.20,aim_weight=0.),
    dict(name='center',distance=[3.,3.],azimuth=0.,elevation=0.,radius=.20,aim_weight=4.),
    dict(name='narrow',distance=[2.5,3.5],azimuth=15.,elevation=10.,radius=.15,aim_weight=4.),
    dict(name='medium',distance=[2.,4.],azimuth=30.,elevation=15.,radius=.10,aim_weight=4.),
    dict(name='wide',distance=[2.,4.],azimuth=60.,elevation=30.,radius=.10,aim_weight=4.),
    dict(name='precise',distance=[2.,4.],azimuth=60.,elevation=30.,radius=.05,aim_weight=4.),
    dict(name='switching',distance=[2.,4.],azimuth=60.,elevation=30.,radius=.05,aim_weight=4.),
]


def sample(level,n,device,generator=None):
    s=STAGES[level];u=torch.rand(n,3,device=device,generator=generator)
    params=torch.stack((s['distance'][0]+u[:,0]*(s['distance'][1]-s['distance'][0]),
                        (2*u[:,1]-1)*math.radians(s['azimuth']),
                        (2*u[:,2]-1)*math.radians(s['elevation']),
                        torch.full_like(u[:,0],s['radius'])),-1)
    return params,s['aim_weight']


def grid(level,device):
    s=STAGES[level]
    distances=sorted(set([s['distance'][0],sum(s['distance'])/2,s['distance'][1]]))
    angle=lambda degrees:sorted(set(math.radians(degrees)*x for x in (-1.,-.5,0.,.5,1.)))
    return torch.tensor([[d,a,e,s['radius']] for d,a,e in itertools.product(distances,angle(s['azimuth']),angle(s['elevation']))],device=device)
