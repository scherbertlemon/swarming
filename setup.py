from setuptools import setup, find_packages
from pathlib import Path
import re


def get_requirements():
    req_file = Path(__file__).parent / "requirements.txt"
    with open(req_file, "r") as req:
        reqs = [r.strip() for r in re.split("\r?\n", req.read()) if r]
    return reqs


setup(
    name="swarming",
    version="0.1",
    description="exploring systems of interacting particles",
    author="Andreas Roth",
    url="https://github.com/scherbertlemon/swarming",
    download_url="https://github.com/scherbertlemon/swarming",
    packages=find_packages(),
    license="MIT License",
    keywords=["swarm", "ode", "particles", "interaction"],
    install_requires=get_requirements(),
    zip_safe=True
)