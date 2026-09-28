"""Process-local compatibility for the pinned mjlab / Warp training stack.

No site-packages files are changed. Only CUDA-graph capability detection is
adapted; physics, observations and optimization are left to the upstream code.
"""
from importlib.metadata import version


def _cuda_graph_supported(simulation):
    import warp as wp
    if not simulation.wp_device.is_cuda:
        return False
    driver = wp.get_cuda_driver_version()
    # Match mjlab 1.2.0's graph gate using Warp 1.13's public driver API.
    supported = driver is not None and driver >= (12, 4)
    supported = supported and wp.is_mempool_enabled(simulation.wp_device)
    if not supported:
        print('[WARNING] CUDA graphs disabled: driver >=12.4 and a memory pool are required')
    return bool(supported)


def configure_training_runtime():
    if version('mjlab') == '1.2.0':
        from mjlab.sim.sim import Simulation
        Simulation._should_use_cuda_graph = _cuda_graph_supported
