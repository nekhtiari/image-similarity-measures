# Releasing

This checklist is for project maintainers. A release must not be published directly from a developer laptop.

## One-time setup

1. Confirm the current maintainers and owners on both GitHub and PyPI.
2. Configure a PyPI Trusted Publisher for the repository's release workflow and protected GitHub environment.
3. Require approval on the release environment.
4. Remove legacy long-lived PyPI tokens only after a Trusted Publishing test succeeds.
5. Enable private vulnerability reporting and branch protection or rulesets on GitHub.

Never add a PyPI password, API token, or generated credential file to this repository.

## Release candidate

1. Review every change since the previous tag and update `CHANGELOG.md`.
2. Remove the `Unreleased` marker from the target version and set its release date.
3. Confirm the version in `pyproject.toml` and refresh `uv.lock`.
4. Run the full CI matrix, security workflow, and minimum-dependency job from a pull request.
5. Build the wheel and source distribution once in CI.
6. Validate metadata and contents with `twine check --strict` and `check-wheel-contents`.
7. Install the wheel with all extras in a clean environment and run the CLI smoke test outside the checkout.
8. Compare compatibility fixtures with the previous release and explicitly review every changed result.

## Publishing

1. Create a signed `vX.Y.Z` tag from the reviewed commit.
2. Let the protected GitHub Actions release job publish the previously validated artifacts through Trusted Publishing.
3. Verify the PyPI metadata, hashes, extras, and console entry point in a new clean environment.
4. Create the GitHub release from the corresponding changelog section.
5. If verification fails, stop and publish a corrective release; PyPI files cannot be replaced.
