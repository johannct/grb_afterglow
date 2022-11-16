from distutils.core import setup
from Cython.Build import cythonize
import numpy

install_requires = [
    "numpy",
    "scipy",
    "cython",
]

setup(
    ext_modules = cythonize(["src/funcs.pyx",
                             "src/takahashi.pyx",]),
    include_dirs=[numpy.get_include()]
)
