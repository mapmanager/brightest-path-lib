# brightest-path-lib

[![Python](https://img.shields.io/badge/python-3.9|3.10|3.11-blue.svg)](https://www.python.org/downloads/release/python-3111/)
[![tests](https://github.com/mapmanager/brightest-path-lib/workflows/Test/badge.svg)](https://github.com/mapmanager/brightest-path-lib/actions)
[![codecov](https://codecov.io/github/mapmanager/brightest-path-lib/branch/main/graph/badge.svg?token=0ZR226588I)](https://codecov.io/github/mapmanager/brightest-path-lib)
[![OS](https://img.shields.io/badge/OS-Linux|Windows|macOS-blue.svg)]()
[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://github.com/mapmanager/brightest-path-lib/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/brightest-path-lib.svg)](https://pypi.org/project/brightest-path-lib/)
<!-- [![Changelog](https://img.shields.io/github/v/release/mapmanager/brightest-path-lib?include_prereleases&label=changelog)](https://github.com/mapmanager/brightest-path-lib/releases) -->

A Python package of path-finding algorithms to find the brightest path between points in an image.

## Getting Started

To install and get started with the `brightest-path-lib`, please visit the [documentation website](https://mapmanager.github.io/brightest-path-lib).

For more detailed instructions on how to use the `brightest-path-lib`, please see the [API documentation](https://mapmanager.github.io/brightest-path-lib/api_docs/).

## Contributing

Contributions are very welcome. Tests can be run with `pytest`, please ensure the coverage at least stays the same before you submit a pull request.

To contribute to this package, first checkout the code. Then create a new virtual environment:

 - With venv
    python -m venv brightest-env
    source brightest-env/bin/activate
 - With conda
    conda create -y -n brightest-env python=3.9
    conda activate brightest-env
    
Now install the package with the testing dependencies:

    cd brightest-path-lib
    pip install -e '.[test]'

To run the tests:

    pytest

## Issues

If you have any suggestions or encounter any problems, please file an [issue](https://github.com/mapmanager/brightest-path-lib/issues) along with a detailed description.

## License

Distributed under the terms of the GNU GPL v3.0 license, "brightest-path-lib" is free and open source software.


