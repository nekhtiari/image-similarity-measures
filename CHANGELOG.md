# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - Unreleased

### Added

- Python 3.10 through 3.14 support and cross-platform CI coverage.
- Public API, CLI, image-loader, legacy edge-case, and installed-artifact tests.
- Branch coverage enforcement, Ruff checks, dependency auditing, CodeQL, dependency review, and Dependabot configuration.
- Correct wheel metadata for the `rasterio` and `speedups` optional dependencies.
- The documented `image-similarity-measures` console entry point in built distributions.
- Explicit OpenCV and Rasterio selection through `read_image_with_loader()`, `evaluation_with_loader()`, and the CLI `--loader` option.
- Contribution, security, compatibility, release, issue, and pull-request guidance.

### Changed

- Migrated project and lockfile management from Poetry to uv.
- Adopted a standard `src/` package layout without changing installed import paths.
- Updated supported dependency ranges and verified their declared minimum versions.
- Updated package ownership and project URL metadata following the repository transfer from UP42.
- Included the full MIT license in built wheels and source distributions.

### Removed

- Python 3.8 and 3.9 support.

### Compatibility note

- Metric implementations and their numerical behavior remain compatible with 0.3.6. Known scientific edge cases are documented but intentionally not corrected in this release.

## [0.3.6] - 2023-05-24

- Last release published before the 0.4 packaging and maintenance modernization.

[Unreleased]: https://github.com/nekhtiari/image-similarity-measures/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/nekhtiari/image-similarity-measures/compare/v0.3.6...v0.4.0
[0.3.6]: https://github.com/nekhtiari/image-similarity-measures/releases/tag/v0.3.6
