"""
setup.py
"""

import os
import shlex
from codecs import open
from subprocess import check_call
from warnings import warn

from setuptools import find_packages, setup
from setuptools.command.develop import develop


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}".format(e))

        develop.run(self)


here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    readme = f.read()

with open("requirements.txt") as f:
    install_requires = f.readlines()

with open(os.path.join(here, "farms", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split("=")[-1].strip().strip('"').strip("'")

test_requires = ["pytest>=5.2"]
description = "The Fast All-sky Radiation Model for Solar applications (FARMS)"

setup(
    name="NREL-farms",
    version=version,
    description=description,
    long_description=readme,
    author="Grant Buster",
    author_email="grant.buster@nrel.gov",
    url="https://github.com/NREL/farms",
    packages=find_packages(),
    package_dir={"farms": "farms"},
    package_data={
        "farms": ["earth_periodic_terms.csv", "sun_earth_radius_vector.csv"]
    },
    include_package_data=True,
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="farms",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "dev": test_requires + ["flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
