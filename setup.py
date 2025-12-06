"""Setup script for PhishDetect package."""

from setuptools import setup, find_packages

with open("README.MD", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PhishDetect",
    version="1.0.0",
    author="Peter",
    description="A machine learning toolkit for phishing email detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "yacs",
        "einops",
        "numpy",
        "pandas",
        "scikit-learn",
        "transformers",
    ],
    extras_require={
        "wandb": ["wandb"],
        "dev": ["pytest", "black", "flake8"],
    },
)
