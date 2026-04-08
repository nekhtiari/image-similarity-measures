# Developer documentation - Image Similarity Measures

The development installation is necessary if you want to contribute to the image-similarity-measures package, e.g. to 
fix a bug.

Clone the repository and set up a dev environment with uv:

```bash
git clone https://github.com/nekhtiari/image-similarity-measures.git
cd image-similarity-measures
uv sync --all-extras --group test
uv run pytest
```

## Upload new package version to PyPI

```bash
uv build
uv publish --token $PYPI_TOKEN
```
