"""Regression for an unmodified mjlab 1.2.0 wheel with Warp 1.13."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import warp as wp
from mjlab.sim.sim import Simulation
from gpt_iayn_core.task import config


class TrainingCompatibilityTests(unittest.TestCase):
    def test_cuda_graph_selection_uses_public_driver_api(self):
        config(1, play=True, dc=True)
        fake = SimpleNamespace(wp_device=SimpleNamespace(is_cuda=True))
        with patch.object(wp, 'is_mempool_enabled', return_value=True):
            for version, expected in [((13, 0), True), ((12, 3), False), (None, False)]:
                with self.subTest(driver=version), patch.object(wp, 'get_cuda_driver_version', return_value=version):
                    try:
                        actual = Simulation._should_use_cuda_graph(fake)
                    except AttributeError as exc:
                        self.fail(f'Fresh mjlab must not depend on removed Warp internals: {exc}')
                    self.assertIs(actual, expected)

    def test_cpu_selection_does_not_query_cuda(self):
        config(1, play=True, dc=True)
        fake = SimpleNamespace(wp_device=SimpleNamespace(is_cuda=False))
        with patch.object(wp, 'get_cuda_driver_version', side_effect=AssertionError('CPU must not query CUDA')):
            self.assertFalse(Simulation._should_use_cuda_graph(fake))
