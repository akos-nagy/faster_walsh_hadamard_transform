from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy

ext_modules = [
    Pybind11Extension(
        "faster_walsh_hadamard_transform",
        ["src/pybind_wrapper.cpp"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++17"],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name="faster_walsh_hadamard_transform",
    version="0.1.0",
    author="Your Name",
    description="Fast(er) Walsh-Hadamard Transform implemented in C++ with OpenMP",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
