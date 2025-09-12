from transonic.dist import make_backend_files, init_transonic_extensions
from pathlib import Path
import numpy as np
import platform

TRANSONIC_BACKEND = "pythran"

def transonize():
    paths = [
        "brightest_path_lib/cost/reciprocal_transonic.py",
        "brightest_path_lib/heuristic/euclidean_transonic.py"
    ]
    here = Path(__file__).parent.absolute()
    make_backend_files([here / path for path in paths], backend=TRANSONIC_BACKEND)

def create_extensions():
    transonize()

    print(f" Backend: {TRANSONIC_BACKEND}")
    # print(f" Compile args: {compile_args}")

    extensions = init_transonic_extensions(
        "brightest_path_lib",
        backend=TRANSONIC_BACKEND,
        include_dirs=np.get_include(),
        compile_args=("-O3", "-march=native", "-DUSE_XSIMD")
    )
    return extensions