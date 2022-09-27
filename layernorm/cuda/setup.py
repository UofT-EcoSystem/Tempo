from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='layernorm',
      ext_modules=[
          CUDAExtension('layernorm', [
              'layernorm.cpp',
              'layernorm_kernel.cu'
          ],
          extra_compile_args=['-O3'])
      ],
      cmdclass={'build_ext': BuildExtension}
)