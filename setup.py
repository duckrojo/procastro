from setuptools import setup, find_packages

# Integrate Sphinx
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

# Recover long description from the README
with open("README.md", "r") as fh:
    long_description = fh.read()

# Package requirements
requirements = ["astropy", "astroquery==0.3.10", "numpy", "scipy", "pyephem", "matplotlib"]

# Setup script
setup(
    # Currently using test.pipy DO NOT UPLOAD TO pipy until release is consolidated
    name="dataproc",     # Distribution name
    version="0.0.8",           # Version name, MUST be updated after each release or will cause an error
    author="Patricio Rojo",
    author_email="pato@das.uchile.cl",
    description="Data processing framework for handling astronomy data files",
    long_description=long_description,      # Currently using the README.md as description
    long_description_content_type="text/markdown",
    url="https://github.com/duckrojo/dataproc",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,  # Will include files listed in MANIFEST.in
    cmdclass = cmdclass,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3.7',
    )