# Developer documentation - Image Similarity Measures

The development installation is necessary if you want to contribute to the image-similarity-measures package, e.g. to 
fix a bug.

Clone the repository and install the library in editable/system-link mode. We recommend using a virtual environment.

```bash
git clone https://github.com/up42/image-similarity-measures.git
cd image-similarity-measures
poetry install
```

## Upload new package version to PyPI

```bash
poetry publish --build --username $PYPI_USERNAME --password $PYPI_PASSWORD
```
