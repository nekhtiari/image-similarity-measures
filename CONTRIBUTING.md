# Contributing

Thank you for helping improve Image Similarity Measures. Compatibility matters because the package is used in existing research and production workflows.

## Before opening a change

- Search existing issues and pull requests first.
- Open an issue before a large API change, new dependency, or numerical correction.
- Keep numerical corrections separate from refactoring and packaging changes.
- Do not update expected metric values without explaining and independently validating the mathematical reason.

## Local setup

Install [uv](https://docs.astral.sh/uv/), then clone and sync the development environment:

```console
git clone https://github.com/nekhtiari/image-similarity-measures.git
cd image-similarity-measures
uv sync --all-extras --group dev
```

Run the same main checks used by continuous integration:

```console
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
uv run twine check --strict dist/*
uv run check-wheel-contents dist/*.whl
```

Use `uv run ruff format .` to format Python files. The lockfile must be updated and committed when dependency declarations change.

## Compatibility requirements

A pull request that changes code should normally include tests. Depending on the change, cover:

- existing public imports and call signatures;
- representative numerical outputs and dtypes;
- channel and shape behavior;
- CLI JSON and exit behavior;
- both OpenCV and Rasterio loading paths;
- installation from the built wheel, not only the source checkout.

Known legacy edge cases are intentionally characterized in the test suite. A test that records an existing defect is not an endorsement of that behavior; it prevents an unrelated refactor from changing results silently. See [COMPATIBILITY.md](COMPATIBILITY.md).

## Pull requests

Keep each pull request focused and describe:

1. the user-visible behavior before and after the change;
2. whether numerical results can change and for which inputs;
3. the tests and external references used to validate the change;
4. any migration or deprecation needed by downstream users.

Do not include credentials, local datasets, generated environments, or built distributions in commits.
