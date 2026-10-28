"""
Setup configuration for Game-Theoretic Modeling of Autonomous Vehicles at Intersections.
"""

from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

# Read requirements
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

# Read long description from README.md
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="av-game-theory",
    version="1.0.0",
    description="Game-Theoretic Modeling and Simulation of Autonomous Vehicles at Intersections",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mohit Agarwal, Ineesh Khindria, Akshit Goel",
    author_email="contact@example.com",
    url="https://github.com/aclox/Game-Theory",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "av-game-demo=demo.run_demo:run_all_experiments",
        ],
    },
)
