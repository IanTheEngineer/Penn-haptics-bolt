from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

# Note: numy.get_include is important to use the correct numpy version (in a non-std location)
#   else we get "RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility"
ext_modules = [Extension("global_align", ["global_align.pyx"], include_dirs=[numpy.get_include()])]

setup(
    name = 'global_align',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)

