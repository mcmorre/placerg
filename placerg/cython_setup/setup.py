#setup.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[Extension('_dice6', ['dice6.pyx'],libraries=['m'], \
    extra_compile_args=["-ffast-math"], include_dirs=[numpy.get_include()])]
setup(
    name='fastloop',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)
