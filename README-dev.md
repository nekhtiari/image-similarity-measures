# Development documentation - Image Similarity Measures

## Installing the required libraries

Follow instructions in [main README](README.md) to setup your virtual environment.

## Install package locally using system link
```bash
pip install -e .
```

## Install `twine`
```bash
pip install twine
```

## Upgrading package in `pypi`

Set appropriate `PYPI_USER` and `PYPI_PASSWORD` in your environment.
```bash
python3 setup.py sdist bdist_wheel
python3 -m twine check dist/*
python3 -m twine upload -u $(PYPI_USER) -p $(PYPI_PASSWORD) dist/*
```
