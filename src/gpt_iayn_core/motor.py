"""Conservative linear motor torque-speed envelope using installed G1 motor specs."""
import re,copy
import numpy as np

def velocity_limit(patterns):
    joined=' '.join(patterns)
    if 'knee' in joined:return 20.
    if 'hip_pitch' in joined:return 32.
    if 'wrist_pitch' in joined:return 22.
    return 37.

def dc_robot_cfg():
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.actuator import DcMotorActuatorCfg
    cfg=copy.deepcopy(get_g1_robot_cfg())
    cfg.articulation.actuators=tuple(DcMotorActuatorCfg(target_names_expr=a.target_names_expr,stiffness=a.stiffness,
        damping=a.damping,effort_limit=a.effort_limit,saturation_effort=a.effort_limit,
        velocity_limit=velocity_limit(a.target_names_expr),armature=a.armature) for a in cfg.articulation.actuators)
    return cfg

def parameters(joint_names):
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    result={k:[] for k in ['kp','kd','effort','velocity']}
    for name in joint_names:
        group=next(a for a in get_g1_robot_cfg().articulation.actuators if any(re.fullmatch(p,name) for p in a.target_names_expr))
        for key,val in [('kp',group.stiffness),('kd',group.damping),('effort',group.effort_limit),('velocity',velocity_limit(group.target_names_expr))]:result[key].append(float(val))
    return result

def torque_numpy(target,q,v,params):
    target=np.asarray(target);q=np.asarray(q);v=np.asarray(v)
    kp=np.asarray(params['kp']);kd=np.asarray(params['kd']);effort=np.asarray(params['effort']);vmax=np.asarray(params['velocity'])
    vel=np.clip(v,-2*vmax,2*vmax)
    top=np.minimum(effort*(1-vel/vmax),effort);bottom=np.maximum(effort*(-1-vel/vmax),-effort)
    return np.clip(kp*(target-q)-kd*v,bottom,top)

def strict_limits(spec):
    for joint in spec.joints:
        if joint.type!=0:
            joint.solref_limit=[.005,1.]
            joint.solimp_limit=[.99,.999,.001,.5,2.]
