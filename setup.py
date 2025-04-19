import sys
from pathlib import Path
from setuptools import setup, find_packages, Extension
from transonic.dist import ParallelBuildExt
# from brightest_path_lib.create_extensions import create_extensions  # Move your functions here

here = Path(__file__).parent.absolute()
sys.path.insert(0, ".")

from create_extensions import create_extensions

setup(
    cmdclass={"build_ext": ParallelBuildExt},
    packages=find_packages(),
    ext_modules=create_extensions(),
)