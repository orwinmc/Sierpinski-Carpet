from distutils.core import setup
from Cython.Build import cythonize

setup(name='Sierpinski Carpet + Graph Approximation',
      ext_modules=cythonize("plus.pyx"))
