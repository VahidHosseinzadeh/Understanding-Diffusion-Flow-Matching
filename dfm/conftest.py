"""Let pytest import the modules, which live in this folder.

Everything is flat inside dfm/: each module imports its siblings by
plain name (`from paths import LinearPath`), exactly as it does when you
run `python dfm/train.py` -- Python puts a script's own folder on the
path. There is no package and nothing to install; this file only tells
pytest the same thing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
