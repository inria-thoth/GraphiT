import sys
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

if sys.platform == 'win32':
    vc_version = os.getenv('VCToolsVersion', '')
    if vc_version.startswith('14.16.'):
        CXX_FLAGS = ['/sdl']
    else:
        CXX_FLAGS = ['/sdl', '/permissive-']
else:
    CXX_FLAGS = ['-g']
    if sys.platform == 'darwin':
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.11'
    else:
        CXX_FLAGS.append('-fopenmp')
    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"

ext_modules = [
    CppExtension(
        'gckn.dynamic_pooling.pooling_cpu', ['gckn/dynamic_pooling/pooling.cpp'],
        extra_compile_args=CXX_FLAGS),
    CppExtension(
        'gckn.gckn_fast.gckn_fast_cpu', ['gckn/gckn_fast/gckn_fast.cpp'],
        extra_compile_args=CXX_FLAGS)
]

# if torch.cuda.is_available() and CUDA_HOME is not None:
#     print(CUDA_HOME)
#     extension = [
#     CUDAExtension(
#         'gckn.dynamic_pooling.pooling_cuda', [
#             'gckn/dynamic_pooling/pooling_cuda.cpp',
#             'gckn/dynamic_pooling/pooling_cuda_kernel.cu'
#         ],
#         extra_compile_args={'cxx': CXX_FLAGS,
#                             'nvcc': ['-O2']}),
#     CUDAExtension(
#         'gckn.gckn_fast.gckn_fast_cuda', [
#             'gckn/gckn_fast/gckn_fast_cuda.cpp',
#             'gckn/gckn_fast/gckn_fast_cuda_kernel.cu'
#         ],
#         extra_compile_args={'cxx': CXX_FLAGS,
#                             'nvcc': ['-O2']})
#     ]
#     ext_modules.extend(extension)

setup(
    name='gckn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
