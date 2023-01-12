from pathlib import Path
from setuptools import setup, find_packages

parent_dir = Path(__file__).resolve().parent

setup(
    name="image-similarity-measures",
    version="0.3.5",
    author="UP42",
    author_email="support@up42.com",
    description="Evaluation metrics to assess the similarity between two images.",
    long_description=parent_dir.joinpath("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/up42/image-similarity-measures",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable",
    ],
    extras_require={
        "rasterio": ["rasterio"],
        "speedups": ["pyfftw"],
    },
    install_requires=parent_dir.joinpath("requirements.txt").read_text().splitlines(),
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "image-similarity-measures=image_similarity_measures.evaluate:main"
        ],
    },
)
