from setuptools import setup, find_packages

VERSION = "0.1"

setup(
    name="brightest-path-lib",
    description="A library of path-finding algorithms to find the brightest path between two points.",
    author="Vasudha Jha",
    url="https://github.com/mapmanager/brightest-path-lib",
    project_urls={
        "Issues": "https://github.com/mapmanager/brightest-path-lib/issues",
        "CI": "https://github.com/mapmanager/brightest-path-lib/actions",
        "Changelog": "https://github.com/mapmanager/brightest-path-lib/releases",
    },
    license="GNU General Public License, Version 3",
    version=VERSION,
    #packages=["brightest_path_lib"],
    #packages=find_packages(),
    packages=find_packages(include=[
        "brightest_path_lib",
        "brightest_path_lib.*",
        "brightest_path_lib.algorithm",
        "brightest_path_lib.input",
        "brightest_path_lib.thread"
        ]),
    install_requires=["numpy"],
    extras_require={"test": ["pytest", "pytest-cov", "scikit-image", "pooch"]},
    python_requires=">=3.7",
)
