"""CPU path-traced review camera for the exact furnished MuJoCo scene. Solver RGB is unchanged."""
import argparse
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import drjit as dr
import mitsuba as mi
import numpy as np
from PIL import Image
mi.set_variant("llvm_ad_rgb")
dr.set_thread_count(3)
T=mi.ScalarTransform4f


def color(rgb):
    a=np.asarray(rgb,dtype=float)
    return (np.maximum(a+.055,0)/1.055)**2.4


def render(args):
    sys.path.insert(0,str(args.task/"tests"))
    import mujoco as mj
    from cameras import Cameras, PAINT
    from interior import finish,furniture_atlas,print_texture
    from world import Home
    w=Home(args.seed)
    cam=Cameras(w,overview=True)
    try:
        cam.sync()
        model,data=cam.model,cam.data
        room=args.room
        body=model.body("display_"+room)
        center=data.xpos[body.id].copy()
        inward=-data.xmat[body.id].reshape(3,3)[:,1]
        lateral=data.xmat[body.id].reshape(3,3)[:,0]
        floor=[(x,-y,1.47) for (x,y),r in w.area.items()
               if r==room and (x,y) not in w.furniture and not w.blocked((x,y))]
        eye=np.asarray(max(floor,key=lambda p: float((p-center)@inward)+.40*abs(float((p-center)@lateral))))
        aim=center+inward*1.25; aim[2]=.91
        scene={"type":"scene","path":{"type":"path","max_depth":7},
               "camera":{"type":"perspective","to_world":T.look_at(origin=eye,target=aim,up=[0,0,1]),
                         "fov":70,"fov_axis":"x","sampler":{"type":"independent","sample_count":args.spp},
                         "film":{"type":"hdrfilm","width":args.width,"height":round(args.width*9/16),
                                 "rfilter":{"type":"tent"}}},
               "outside":{"type":"constant","radiance":{"type":"rgb","value":[.50,.59,.68]}}}
        names={model.mat(i).name for i in range(model.nmat)}
        with TemporaryDirectory() as td:
            dest=Path(td)
            for n in names:
                if n.startswith("finish_"): buf=finish(n[7:],PAINT[n[7:]])
                elif n=="plaster":buf=finish("plaster","dfdbcf")
                elif n=="textile":buf=finish("runner",PAINT["runner"])
                elif n=="joinery":buf=furniture_atlas()
                elif n.startswith("art") and n[3:].isdigit():buf=print_texture(int(n[3:]))
                else: continue
                (dest/(n+".png")).write_bytes(buf)
            for i in range(model.nmesh):
                v0,nv=model.mesh_vertadr[i],model.mesh_vertnum[i]
                f0,nf=model.mesh_faceadr[i],model.mesh_facenum[i]
                verts=model.mesh_vert[v0:v0+nv]; faces=model.mesh_face[f0:f0+nf]
                ta,nt=model.mesh_texcoordadr[i],model.mesh_texcoordnum[i]
                textured=ta>=0 and nt>0
                lines=["v %.6f %.6f %.6f"%tuple(v) for v in verts]
                if textured:
                    lines.extend("vt %.6f %.6f"%tuple(v) for v in model.mesh_texcoord[ta:ta+nt])
                for j,f in enumerate(faces):
                    uv=model.mesh_facetexcoord[f0+j]
                    lines.append("f "+" ".join(f"{a+1}/{b+1}" if textured and b>=0 else str(a+1) for a,b in zip(f,uv)))
                (dest/f"mesh{i}.obj").write_text("\n".join(lines))
            for i in range(model.ngeom):
                if model.geom_rgba[i,3]<.01: continue
                mat_id=model.geom_matid[i]
                name=model.mat(mat_id).name if mat_id>=0 else ""
                typ=model.geom_type[i]
                size=model.geom_size[i]; xyz=data.geom_xpos[i]
                transform=np.eye(4);transform[:3,:3]=data.geom_xmat[i].reshape(3,3);transform[:3,3]=xyz
                tr=T(transform.tolist())
                if typ==mj.mjtGeom.mjGEOM_BOX:
                    shape={"type":"cube","to_world":tr@T.scale(size.tolist())}
                elif typ==mj.mjtGeom.mjGEOM_MESH:
                    shape={"type":"obj","filename":str(dest/f"mesh{model.geom_dataid[i]}.obj"),"to_world":tr,"face_normals":True,"flip_tex_coords":False}
                elif typ==mj.mjtGeom.mjGEOM_CYLINDER:
                    shape={"type":"cylinder","p0":[0,0,-size[1]],"p1":[0,0,size[1]],"radius":size[0],"to_world":tr}
                elif typ in (mj.mjtGeom.mjGEOM_SPHERE,mj.mjtGeom.mjGEOM_ELLIPSOID):
                    shape={"type":"sphere","to_world":tr@T.scale((size if typ==mj.mjtGeom.mjGEOM_ELLIPSOID else [size[0]]*3))}
                else: continue
                if name and (dest/(name+".png")).exists():
                    tex={"type":"bitmap","filename":str(dest/(name+".png")),"filter_type":"bilinear"}
                    if name.startswith("finish_"):
                        tex["to_uv"]=mi.ScalarTransform3f().scale([size[0]*.86,size[1]*.86])
                    elif name=="plaster":
                        tex["to_uv"]=mi.ScalarTransform3f().scale([max(size[:2])*1.5,size[2]*1.5])
                    albedo=tex
                else:
                    a=model.mat_rgba[mat_id,:3] if name in ("glazing","ceilingpaint") else model.geom_rgba[i,:3]
                    albedo={"type":"rgb","value":color(a).clip(.02,.92).tolist()}
                shape["bsdf"]={"type":"diffuse","reflectance":albedo}
                if name.startswith("finish_") or name=="joinery":
                    shape["bsdf"]={"type":"roughplastic","alpha":.32,"int_ior":1.38,"diffuse_reflectance":albedo}
                if name=="glazing":
                    shape["emitter"]={"type":"area","sampling_weight":float(1/(1+np.linalg.norm(xyz-eye)**2)),
                                      "radiance":{"type":"rgb","value":[4.6,5.2,5.8]}}
                scene["g"+str(i)]=shape
            # Soft interior fill as from a ceiling luminaire, plus daylight through physical window panes.
            xs=[p[0] for p in floor]; ys=[p[1] for p in floor]
            for j,(u,v) in enumerate(((.33,.46),(.72,.72))):
                p=[min(xs)+(max(xs)-min(xs))*u,min(ys)+(max(ys)-min(ys))*v,2.445]
                scene[f"ceiling_fill{j}"]={"type":"rectangle","to_world":T.translate(p)@T.rotate([1,0,0],180)@T.scale([.75,.52,1]),
                                      "emitter":{"type":"area","sampling_weight":2.0,"radiance":{"type":"rgb","value":[6.8,6.3,5.6]}}}
            s=mi.load_dict(scene)
            img=mi.render(s,spp=args.spp,seed=17)
            mi.util.write_bitmap(str(args.output),img)
            dr.sync_thread()
    finally:
        cam.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--task",type=Path,required=True)
    p.add_argument("--seed",type=int,default=804)
    p.add_argument("--room",default="slate")
    p.add_argument("--width",type=int,default=960)
    p.add_argument("--spp",type=int,default=32)
    p.add_argument("--output",type=Path,required=True)
    render(p.parse_args())
