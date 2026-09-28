"""Locations in the editable source checkout, independent of the process cwd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE_SOURCE = Path(__file__).with_name('acceptance.py')
