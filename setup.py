from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(name='tempo',
      ext_modules=[
            CppExtension(name='tempo.backend.combined', sources=['src/combined/combined.cpp'], extra_compile_args=['-Ofast']),
            CUDAExtension('tempo.backend.inplace_gelu', ['src/inplace_gelu/gelu.cpp', 'src/inplace_gelu/gelu_kernel.cu'], extra_compile_args=['-O3']),
            CUDAExtension('tempo.backend.inplace_layernorm', ['src/inplace_layernorm/inplace_layernorm.cpp', 'src/inplace_layernorm/inplace_layernorm_kernel.cu'], extra_compile_args=['-O3'])
        ],
      packages=["tempo"],
      cmdclass={'build_ext': BuildExtension}
)

