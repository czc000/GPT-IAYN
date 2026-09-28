"""Native APIs remain usable outside the repository working directory."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTests(unittest.TestCase):
    def test_native_imports_and_paths_from_unrelated_directory(self):
        code = '''
import sys, importlib.abc, pathlib
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('mjlab', 'rsl_rl', 'warp'):
   raise RuntimeError('Native execution imported a training dependency: '+fullname)
sys.meta_path.insert(0, Block())
from gpt_iayn_core import evaluate_rl, observations, policy_export
from gpt_iayn_core.paths import ROOT, ACCEPTANCE_SOURCE
assert (ROOT/'release-data/SHA256SUMS').is_file()
assert ACCEPTANCE_SOURCE.is_file()
assert observations.native_obs.__module__ == 'gpt_iayn_core.observations'
'''
        env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
        with tempfile.TemporaryDirectory(prefix='gpt-iayn-import-') as directory:
            result = subprocess.run([sys.executable, '-c', code], cwd=directory, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
