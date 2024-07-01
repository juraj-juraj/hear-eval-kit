#!/usr/bin/env python3

import os

# Always prefer setuptools over distutils
import sys

from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="heareval",
    description="Holistic Evaluation of Audio Representations (HEAR) 2021 -- Evaluation Kit",
    author="",
    author_email="",
    url="https://github.com/hearbenchmark/hear-eval-kit",
    download_url="https://github.com/hearbenchmark/hear-eval-kit",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/hearbenchmark/hear-eval-kit/issues",
        "Source Code": "https://github.com/hearbenchmark/hear-eval-kit",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=[
        "click",
        "dcase_util",
        "intervaltree",
        "more-itertools",
        "numpy>=1.24.3",
        "pandas",
        "pynvml",
        "pytorch-lightning>=2.0",
        "python-slugify",
        "sed_eval",
        "soundfile",
        "spotty",
        "torch>=2.0",
        "torchinfo",
        "tqdm",
        # "wandb",
        "scikit-learn>=0.24.2",
        "submitit",
        # otherwise librosa breaks, which fucking dcase-util requires
        # "numba",
        # "numba>=0.49.0",  # not directly required, pinned by Snyk to avoid a vulnerability
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
    },
    classifiers=[],
)
