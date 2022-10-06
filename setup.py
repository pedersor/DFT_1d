from setuptools import find_packages, setup

with open("README.md", "r") as fh:
  long_description = fh.read()

# Read in requirements
requirements = [
    requirement.strip() for requirement in open('requirements.txt').readlines()
]

setup(
    name="DFT_1d",
    version="0.0.1",
    author="Ryan Pederson",
    author_email="pedersor@uci.edu",
    description="A python based 1D DFT solver.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pedersor/DFT_1d",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=('>=3.6.9'),
    install_requires=requirements,
    packages=find_packages(),
)
