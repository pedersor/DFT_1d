#THIS IS FOR IF YOU DO A PYPI

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="DFT_1d",  # Replace with your own username
    version="0.0.0",
    author="R.P. etc...",
    author_email="pedersor@uci.edu",
    description="This is a python based 1D solver.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="URL_TO_REPO",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",  #TALK TO RYAN M IF YOU WANT LISCENCE ADVICE
        "Operating System :: OS Independent",  #IF fully python
        "Development Status :: 4 - Beta",  #IF actually beta at that point
    ],
    python_requires='>=3.6',  #??
)