# several files with ext .pyx, that i will call by their name
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("plus",       ["plus.pyx"]),
    Extension("shared",         ["shared.pyx"])
]

setup(
  name = 'Sierpinski Carpet + Graph Approximations',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
)
