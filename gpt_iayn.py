"""Compatibility launcher: python gpt_iayn.py --help (no editable install required)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from gpt_iayn_core.cli import main

if __name__ == '__main__':
    main()
