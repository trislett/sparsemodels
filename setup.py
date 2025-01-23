import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

PACKAGE_NAME = "sparsemodels"
BUILD_REQUIRES = [
    "numpy", "scipy", "statsmodels", "argparse",
    "matplotlib", "pandas", "nibabel", "cython",
    "scikit-learn", "scikit-image", "rpy2",
    "seaborn", "tqdm"
]

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering :: Medical Science Apps."
]

# Define Cython extensions
extensions = [
    Extension(
        "sparsemodels.cynumstats",  # Module name
        sources=["sparsemodels/cynumstats.pyx"],  # Cython source
        include_dirs=[numpy.get_include()],  # Include NumPy headers
        language="c"
    )
]

setup(
    name=PACKAGE_NAME,
    version="0.0.1",
    include_package_data=True,
    maintainer="Tristram Lett",
    maintainer_email="tristram.lett@charite.de",
    description="Sparse (mostly) statistical functions",
    long_description="Sparse statistical functions for scientific analysis.",
    url="https://github.com/trislett/sparsemodels.git",
    classifiers=CLASSIFIERS,
    zip_safe=False,
    install_requires=BUILD_REQUIRES,
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level="3"),
    python_requires=">=3.7",
)


