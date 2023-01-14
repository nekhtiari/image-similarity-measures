# Developer documentation - Image Similarity Measures

The development installation is necessary if you want to contribute to the image-similarity-measures package, e.g. to 
fix a bug.

Clone the repository and install the library in editable/system-link mode. We recommend using a virtual environment.

```bash
git clone https://github.com/up42/image-similarity-measures.git
cd image-similarity-measures
pip install -e .
```

## Upload new package version to PyPI

```bash
pip install twine
```

Install the [twine package](https://pypi.org/project/twine/) and set the `TWINE_USERNAME` and `TWINE_PASSWORD` 
environment variables with your PyPI credentials.

```bash
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
```

