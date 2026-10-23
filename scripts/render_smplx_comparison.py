"""Render synchronized SMPL-X mesh / UMR robot using UMR's mesh utilities."""
import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/UMR/scripts'))
sys.path.insert(0, str(ROOT / 'third_party/UMR'))
import smpl_surface_retarget_common as common
from view_smpl_mujoco import write_obj, update_dynamic_mesh


def scene_setup(root):
    world = root.find('worldbody')
    for plane in list(world.findall('geom[@type="plane"]')):
        world.remove(plane)
    ET.SubElement(world, 'geom', type='plane', size='20 20 .01', rgba='.87 .89 .92 1', contype='0', conaffinity='0')
    ET.SubElement(world, 'light', pos='0 -3 5', dir='0 1 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'light', pos='0 3 4', dir='0 -1 -1', diffuse='.7 .7 .7')
    asset = root.find('asset')
    if asset is None:asset = ET.SubElement(root, 'asset')
    for sky in list(asset.findall('texture[@type="skybox"]')):asset.remove(sky)
    ET.SubElement(asset, 'texture', name='comparison_sky', type='skybox', builtin='flat', rgb1='.95 .96 .98', width='128', height='128')


def panel(renderer, model, data, azimuth):
    cam = mj.MjvCamera()
    cam.lookat[:] = [0, 0, 1.25]
    cam.distance, cam.azimuth, cam.elevation = 4, azimuth, 0
    renderer.update_scene(data, camera=cam)
    for camera in renderer.scene.camera:
        camera.orthographic = 1
        camera.frustum_bottom, camera.frustum_top = -1.25, 1.25
        camera.frustum_near, camera.frustum_far = .01, 50
    for z in np.arange(0, 2.51, .5):
        scene = renderer.scene
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([.55,.58,.63,1]))
        start = [-1.25,-.8,z] if azimuth==90 else [-.8,-1.25,z]
        end = [1.25,-.8,z] if azimuth==90 else [-.8,1.25,z]
        mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_LINE,1,np.array(start),np.array(end))
        scene.ngeom+=1
    return renderer.render().copy()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=300)
    p.add_argument('--frames',type=int,default=600)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.result,allow_pickle=False) as d:
        q=d['qpos'][a.start:a.start+a.frames].copy()
        frame_ids=d['frame_ids'][a.start:a.start+a.frames].copy()
        fps=float(d['fps'].item());scale=float(d['smpl_scale'].item());ground=float(d['ground_z'].item())
        name=str(d['robot_name'].item());seq=str(d['source_sequence_key'].item())
        source=Path(str(d['source_data'].item())) / (seq+'.npz')
        robot_xml=Path(str(d['robot_xml'].item()))
    if len(q)!=a.frames:raise ValueError('Requested window exceeds result')
    collection,_=common.load_motion_collection(source)
    sequence=collection[seq]
    vertices,joints,faces=common.source_motion_vertices_joints(sequence,frame_ids,a.model_dir,batch_size=32,smplx_device='cpu',smplx_batch_size=32)
    vertices=common.source_points_to_retarget_frame(vertices,'smplx',sequence.get('output_up','z'))
    joints=common.source_points_to_retarget_frame(joints,'smplx',sequence.get('output_up','z'))
    vertices[:,:,2]-=ground
    vertices*=scale
    vertices[:,:,:2]-=joints[:,:1,:2]*scale
    q[:,:2]=0
    tree=ET.parse(robot_xml);scene_setup(tree.getroot())
    robot=mj.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))
    rd=mj.MjData(robot)
    with tempfile.TemporaryDirectory() as temp:
        mesh=Path(temp)/'human.obj';write_obj(mesh,vertices[0],faces)
        root=ET.fromstring(f'<mujoco><asset><mesh name="human" file="{mesh}"/></asset><worldbody><geom type="mesh" mesh="human" rgba=".18 .48 .76 1" contype="0" conaffinity="0"/></worldbody></mujoco>')
        scene_setup(root)
        human=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));hd=mj.MjData(human)
        for model in (human,robot):model.vis.global_.offwidth=480;model.vis.global_.offheight=480
        fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        font=ImageFont.truetype(fontpath,18) if Path(fontpath).exists() else ImageFont.load_default()
        snapshots=[];chosen=set(np.linspace(0,len(q)-1,4,dtype=int))
        with mj.Renderer(human,480,480) as hr,mj.Renderer(robot,480,480) as rr,imageio.get_writer(a.output/'comparison.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=16) as writer:
            for t,pose in enumerate(q):
                update_dynamic_mesh(human,0,vertices[t],faces)
                # Each renderer owns its own GL context.
                hr._gl_context.make_current()
                mj.mjr_uploadMesh(human,hr._mjr_context,0)
                mj.mj_forward(human,hd)
                rd.qpos[:]=pose;mj.mj_forward(robot,rd)
                rows=[]
                for azimuth in (90,0):
                    hr._gl_context.make_current();left=panel(hr,human,hd,azimuth)
                    rr._gl_context.make_current();right=panel(rr,robot,rd,azimuth)
                    rows.append(np.concatenate([left,right],axis=1))
                canvas=Image.new('RGB',(960,1040),'#f2f5f8');canvas.paste(Image.fromarray(np.concatenate(rows)),(0,80))
                draw=ImageDraw.Draw(canvas)
                draw.text((15,8),f'{seq} | {frame_ids[t]/fps:.2f}s | kinematic retarget replay',fill='#17232f',font=font)
                draw.text((15,35),f'SMPL-X (retarget scale {scale:.3f})',fill='#245a88',font=font)
                draw.text((495,35),name,fill='#17232f',font=font)
                draw.text((15,59),'Front / side | root-centred | 2.5 m frame | grid 0.5 m',fill='#526172',font=font)
                frame=np.asarray(canvas);writer.append_data(frame)
                if t in chosen:snapshots.append(frame)
                if t%100==0:print(f'{name}: {t}/{len(q)}',flush=True)
        imageio.imwrite(a.output/'contact_sheet.png',np.concatenate(snapshots,axis=0))
    (a.output/'render.json').write_text(json.dumps({'result':str(a.result),'source':str(source),'robot_xml':str(robot_xml),'source_frame_ids':[int(frame_ids[0]),int(frame_ids[-1])],'frames':len(q),'fps':fps,'human_scale':scale,'ground_z':ground,'views':['front','side'],'root_xy_centered':True,'render':'SMPL-X mesh and robot visual geometry; kinematic replay'},indent=2))


if __name__=='__main__':main()
