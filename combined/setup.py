from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='combined',
      ext_modules=[cpp_extension.CppExtension(name='combined_cpp', 
                                              sources=['combined.cpp'],
                                              extra_compile_args=['-Ofast'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
