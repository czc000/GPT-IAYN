"""Native evaluation helpers; no training framework or CUDA imports."""
from pathlib import Path
import mujoco


def motion_path(run, contract):
    path = Path(contract['motion_file'])
    return path if path.is_absolute() else Path(run) / path


def apply_physics(model, cfg):
    """Apply the serialized experiment options without depending on mjlab."""
    enums = {
        'integrator': ('mjtIntegrator', 'mjINT_'),
        'cone': ('mjtCone', 'mjCONE_'),
        'jacobian': ('mjtJacobian', 'mjJAC_'),
        'solver': ('mjtSolver', 'mjSOL_'),
    }
    defaults = dict(timestep=.002, integrator='implicitfast', impratio=1.,
                    cone='pyramidal', jacobian='auto', solver='newton',
                    iterations=100, tolerance=1e-8, ls_iterations=50,
                    ls_tolerance=.01, ccd_iterations=50, gravity=[0,0,-9.81], multiccd=False)
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown physics options: {sorted(unknown)}')
    for key,value in (defaults | cfg).items():
        if key in enums:
            group,prefix = enums[key]
            value = getattr(getattr(mujoco,group), prefix+value.upper())
        if key == 'gravity':
            model.opt.gravity[:] = value
        elif key == 'multiccd':
            # Match the historical mjlab apply(): false leaves XML/defaults alone.
            # New MuJoCo versions removed the old opt-in MULTICCD enum entirely.
            if value:
                bit = getattr(mujoco.mjtEnableBit,'mjENBL_MULTICCD',None)
                if bit is None:raise ValueError('This MuJoCo version has no legacy MULTICCD opt-in flag')
                model.opt.enableflags |= int(bit)
        else:
            setattr(model.opt,key,value)


def video_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype('DejaVuSans.ttf',size)
    except OSError:
        return ImageFont.load_default(size=size)
