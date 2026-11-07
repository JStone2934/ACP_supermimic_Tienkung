"""Shared deterministic surface sampling and fixed-chain transforms for contact geometry."""
from __future__ import annotations
import itertools
import mujoco
import numpy as np
from scipy.spatial import ConvexHull


def quat_mat(q):
    r=np.empty(9); mujoco.mju_quat2Mat(r,np.asarray(q,dtype=np.float64)); return r.reshape(3,3)


def fixed_transform(m, frame, link):
    """Link origin in frame; reject any intervening active joint."""
    p,R=np.zeros(3),np.eye(3)
    while link != frame:
        if link==0 or m.body_jntnum[link]:
            raise ValueError('contact geometry must belong to its frame or a fixed descendant')
        r=quat_mat(m.body_quat[link]); p=m.body_pos[link]+r@p; R=r@R
        link=m.body_parentid[link]
    return p,R


def surface_points(m,g,spacing=.008):
    """Convex mesh/primitive surface grid with bounded edge spacing, in geom coordinates."""
    t,s=m.geom_type[g],m.geom_size[g]
    if t==mujoco.mjtGeom.mjGEOM_MESH:
        i=m.geom_dataid[g]; v=m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i]+m.mesh_vertnum[i]].astype(float)
        hull=ConvexHull(v); tris=v[hull.simplices]
    elif t==mujoco.mjtGeom.mjGEOM_BOX:
        v=np.array(list(itertools.product((-1,1),repeat=3)))*s
        tris=v[ConvexHull(v).simplices]
    else:
        n=max(16,int(np.ceil(2*np.pi*s[0]/spacing)))
        theta=np.linspace(0,2*np.pi,n,endpoint=False)
        if t==mujoco.mjtGeom.mjGEOM_CYLINDER:
            zs=np.linspace(-s[1],s[1],max(2,int(np.ceil(2*s[1]/spacing))+1))
            side=np.array([[s[0]*np.cos(a),s[0]*np.sin(a),z] for a in theta for z in zs])
            rs=np.linspace(0,s[0],max(2,int(np.ceil(s[0]/spacing))+1))
            caps=np.array([[r*np.cos(a),r*np.sin(a),z] for r in rs for a in theta for z in (-s[1],s[1])])
            return np.concatenate((side,caps))
        if t in (mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_CAPSULE):
            phi=np.linspace(0,np.pi,max(9,n//2+1))
            v=np.array([[s[0]*np.sin(p)*np.cos(a),s[0]*np.sin(p)*np.sin(a),s[0]*np.cos(p)] for p in phi for a in theta])
            if t==mujoco.mjtGeom.mjGEOM_CAPSULE:
                v[:,2]+=np.where(v[:,2]>=0,s[1],-s[1])
                zs=np.linspace(-s[1],s[1],max(2,int(np.ceil(2*s[1]/spacing))+1))
                v=np.concatenate((v,np.array([[s[0]*np.cos(a),s[0]*np.sin(a),z] for a in theta for z in zs])))
            return v
        raise ValueError(f'unsupported geom type {t}')
    pts=[]
    for a,b,c in tris:
        n=max(1,int(np.ceil(max(np.linalg.norm(b-a),np.linalg.norm(c-a),np.linalg.norm(c-b))/spacing)))
        pts.extend(a+(i/n)*(b-a)+(j/n)*(c-a) for i in range(n+1) for j in range(n+1-i))
    return np.unique(np.round(np.asarray(pts),9),axis=0)


def hull_data(m,g,p0,R0):
    """Convex hull in the semantic body frame; face support intervals are precomputed."""
    i=m.geom_dataid[g]
    verts=m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i]+m.mesh_vertnum[i]].astype(float)
    h=ConvexHull(verts); verts=verts[h.vertices]
    R=R0@quat_mat(m.geom_quat[g]); p=p0+R0@m.geom_pos[g]
    verts=p+verts@R.T
    h=ConvexHull(verts); normals=h.equations[:,:3]
    # Remove coplanar duplicates without discarding curved faces.
    normals=np.unique(np.round(normals,7),axis=0)
    normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
    lower=[]; upper=[]
    for chunk in np.array_split(normals,max(1,len(normals)//128)):
        projection=chunk@verts.T; lower.extend(projection.min(-1)); upper.extend(projection.max(-1))
    # Cross products of mesh edges and box axes are needed at box corners.
    pairs=np.concatenate((h.simplices[:,[0,1]],h.simplices[:,[1,2]],h.simplices[:,[0,2]]))
    edges=verts[pairs[:,1]]-verts[pairs[:,0]]
    edges=edges[np.linalg.norm(edges,axis=-1)>1e-8]
    edges/=np.linalg.norm(edges,axis=-1,keepdims=True)
    # Directions are unoriented; choose a consistent hemisphere before angular coverage sampling.
    dominant=np.argmax(np.abs(edges),axis=-1)
    edges*=np.sign(edges[np.arange(len(edges)),dominant])[:,None]
    edges=np.unique(np.round(edges,6),axis=0)
    selected=[0]; dist=1-np.abs(edges@edges[0])
    for _ in range(min(256,len(edges))-1):
        j=int(np.argmax(dist)); selected.append(j)
        dist=np.minimum(dist,1-np.abs(edges@edges[j]))
    return dict(vertices=verts,normals=normals,lower=np.array(lower),upper=np.array(upper),edges=edges[selected])


def hull_box_penetration(p,R,box,hull):
    """Separating face/box axes in torch; negative overlap depth.

    Face axes plus up to 256 angularly distributed mesh-edge directions crossed with
    box axes. Accuracy is measured against MuJoCo, not claimed as generic exact GJK.
    """
    import torch
    normals,verts,lower,upper=(hull[k].to(dtype=p.dtype,device=p.device) for k in ('normals','vertices','lower','upper'))
    center=torch.stack((box[:,0],box[:,1],box[:,2]-box[:,7]),-1)
    c,s=box[:,3],box[:,4]; zero=torch.zeros_like(c); one=torch.ones_like(c)
    axes=torch.stack((torch.stack((c,s,zero),-1),torch.stack((-s,c,zero),-1),torch.stack((zero,zero,one),-1)),1)
    # Mesh face axes: mesh support intervals do not depend on pose.
    world_normals=normals@R.transpose(-1,-2)
    delta=center-p
    mid=(world_normals*delta[:,None]).sum(-1)
    radius=(torch.abs(world_normals@axes.transpose(-1,-2))*box[:,None,5:8]).sum(-1)
    sep_face=torch.maximum(lower-mid-radius,mid-radius-upper).amax(-1)
    # Box axes: support all mesh vertices, including vertices absent from surface samples.
    local_axes=axes@R
    proj=local_axes@verts.T
    mid_box=(axes*delta[:,None]).sum(-1)
    sep_box=torch.maximum(proj.amin(-1)-mid_box-box[:,5:8],mid_box-box[:,5:8]-proj.amax(-1)).amax(-1)
    result=torch.maximum(sep_face,sep_box)
    edges=hull['edges'].to(dtype=p.dtype,device=p.device)
    edge_axes=torch.cross(edges[None,:,None,:].expand(len(p),-1,3,-1),local_axes[:,None].expand(-1,len(edges),-1,-1),dim=-1).flatten(1,2)
    norm=edge_axes.norm(dim=-1,keepdim=True)
    edge_axes=edge_axes/norm.clamp(min=1e-8)
    local_delta=(R.transpose(-1,-2)@delta.unsqueeze(-1)).squeeze(-1)
    for start in range(0,edge_axes.shape[1],96):
        ax=edge_axes[:,start:start+96]
        proj=ax@verts.T
        mid=(ax*local_delta[:,None]).sum(-1)
        radius=(torch.abs(ax@local_axes.transpose(-1,-2))*box[:,None,5:8]).sum(-1)
        sep=torch.maximum(proj.amin(-1)-mid-radius,mid-radius-proj.amax(-1))
        sep=torch.where(norm[:,start:start+96,0]>1e-6,sep,-torch.inf)
        result=torch.maximum(result,sep.amax(-1))
    return result
