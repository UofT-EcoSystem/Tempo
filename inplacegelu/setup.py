from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='gelu',
      ext_modules=[
          CUDAExtension('gelu_cuda', [
              'gelu.cpp',
              'gelu_cuda.cu',
          ], 
          extra_compile_args=['-O3'])
      ],
      cmdclass={'build_ext': BuildExtension})
