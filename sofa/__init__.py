from .models.SOFA import SOFA
from . import plots as pl
from . import utils as tl

import toml
from pathlib import Path

def get_version():
   path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
   print(path)
   pyproject = toml.loads(open(str(path)).read())
   return pyproject['tool']['poetry']['version']

#__version__ = get_version()

__all__ = ['pl', 'tl', 'SOFA']
