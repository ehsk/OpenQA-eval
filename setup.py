import os
from setuptools import find_packages, setup
from typing import List

VERSION = {}  # type: ignore
with open("src/oqaeval/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

_PROJECT_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#") -> List[str]:
    """Source: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/setup_tools.py
    Load requirements from a file.
    >>> _load_requirements(_PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


setup(
    name="oqaeval",
    version=VERSION["VERSION"],
    url="https://github.com/ehsk/OpenQA-eval",
    license="MIT",
    author="Ehsan Kamalloo",
    author_email="ekamalloo@uwaterloo.ca",
    description="Evaluating Open-Domain Question Answering using Large Language Models",
    package_dir={"": "src"},
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=_load_requirements(_PROJECT_ROOT),
    python_requires=">=3.8.0",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Natural Language Processing",
        "Topic :: Scientific/Engineering :: Question Answering",
        "Topic :: Scientific/Engineering :: Large Language Models",
    ],
)
