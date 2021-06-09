from distutils.extension import Extension

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('graphs', parent_package, top_path)

    extensions = [
        Extension("gckn.graphs.graphs_fast", 
                ['gckn/graphs/graphs_fast.pyx'],
                extra_compile_args = ["-ffast-math"],
                include_dirs = [numpy.get_include()]
                )
    ]
    

    config.ext_modules += extensions

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
