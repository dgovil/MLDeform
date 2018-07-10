from pkgutil import extend_path
import os
import sys

if isinstance(__path__, str):
    __path__ = [__path__]

# If imported in Maya, add the Maya package inline
if 'maya' in sys.executable.lower():
    __path__.append(os.path.join(__path__[-1], '_maya'))

__path__.append(os.path.join(__path__[-1], '_training'))

# Extend the path in case other locations in the system
# need to add on to this
__path__ = extend_path(__path__, __name__)