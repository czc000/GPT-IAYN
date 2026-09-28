"""Flatten MJCF defaults before importing with Isaac's MJCF importer."""
import json,xml.etree.ElementTree as ET,shutil
from pathlib import Path
import mujoco
from gpt_iayn_core.evaluate import load_native_model
from gpt_iayn_core.scene import ROOT
src=ROOT/'artifacts/final';out=ROOT/'artifacts/rl-isaac/scene';out.mkdir(parents=True,exist_ok=True)
shutil.copytree(src/'scene',out,dirs_exist_ok=True)
c=json.loads((src/'contract.json').read_text());m=load_native_model(src,c)
tree=ET.parse(src/'scene/scene.xml');xml=tree.getroot();fmt=lambda a:' '.join(f'{float(v):.12g}' for v in a)
for body in xml.findall('.//worldbody//body'):
 name=body.get('name');bid=m.body(name).id
 geoms=list(body.findall('geom'));assert len(geoms)==m.body_geomnum[bid]
 for geom,gid in zip(geoms,range(m.body_geomadr[bid],m.body_geomadr[bid]+m.body_geomnum[bid])):
  geom.attrib.pop('class',None);geom.attrib.pop('fromto',None)
  typ=mujoco.mjtGeom(int(m.geom_type[gid])).name.removeprefix('mjGEOM_').lower()
  geom.set('type',typ);geom.set('pos',fmt(m.geom_pos[gid]));geom.set('quat',fmt(m.geom_quat[gid]));geom.set('size',fmt(m.geom_size[gid]))
  geom.set('contype',str(m.geom_contype[gid]));geom.set('conaffinity',str(m.geom_conaffinity[gid]));geom.set('friction',fmt(m.geom_friction[gid]))
  geom.set('rgba',fmt(m.geom_rgba[gid]))
  if typ=='mesh':geom.set('mesh',m.mesh(int(m.geom_dataid[gid])).name)
 inertial=body.find('inertial')
 if inertial is None:inertial=ET.SubElement(body,'inertial')
 inertial.attrib.clear();inertial.set('pos',fmt(m.body_ipos[bid]));inertial.set('quat',fmt(m.body_iquat[bid]));inertial.set('mass',str(m.body_mass[bid]));inertial.set('diaginertia',fmt(m.body_inertia[bid]))
 for joint in body.findall('joint'):
  jid=m.joint(joint.get('name')).id;dof=m.jnt_dofadr[jid]
  joint.set('armature',str(m.dof_armature[dof]));joint.set('damping',str(m.dof_damping[dof]))
world=xml.find('worldbody')
for child in list(world):
 if child.tag!='body' or child.get('name')!='robot/pelvis':world.remove(child)
for tag in ['sensor','contact','equality','keyframe','default']:
 elem=xml.find(tag)
 if elem is not None:xml.remove(elem)
for elem in xml.iter():
 elem.attrib.pop('class',None);elem.attrib.pop('childclass',None)
 for key,value in list(elem.attrib.items()):
  if key!='file' and 'robot/' in value:elem.set(key,value.replace('robot/',''))
tree.write(out/'robot.xml')
meta=dict(joint_armature={n:float(m.dof_armature[m.jnt_dofadr[m.joint('robot/'+n).id]]) for n in c['joint_names']},mass={n:float(m.body_mass[m.body('robot/'+n).id]) for n in c['body_names']})
(ROOT/'artifacts/rl-isaac/native-model.json').write_text(json.dumps(meta,indent=2));print('flattened MJCF and actuator armatures')
