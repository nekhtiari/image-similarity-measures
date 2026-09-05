# Modernization roadmap

This roadmap keeps release engineering, refactoring, numerical corrections, and new features in separate reviewable changes.

## 0.4: compatibility and maintenance

The 0.4 release is intentionally conservative:

- preserve the 0.3.6 metric implementations and default loader behavior;
- restore working wheel extras and the console entry point;
- support Python 3.10 through 3.14;
- add compatibility, minimum-dependency, platform, artifact, and security checks;
- adopt a `src/` layout and add explicit, opt-in loader selection;
- document ownership, security, contribution, compatibility, and releases.

After a public release candidate is tested, packaging reports [#65](https://github.com/nekhtiari/image-similarity-measures/issues/65), [#74](https://github.com/nekhtiari/image-similarity-measures/issues/74), and [#75](https://github.com/nekhtiari/image-similarity-measures/issues/75) should be reproduced against the candidate before they are closed. Dependency-only pull requests [#69](https://github.com/nekhtiari/image-similarity-measures/pull/69) and [#73](https://github.com/nekhtiari/image-similarity-measures/pull/73) can then be reviewed as potentially superseded.

## Next: correctness specifications

Each topic below should begin with reference fixtures and a short written specification. Formula-changing work should not be bundled together.

1. Define accepted shapes, channel order, dtypes, masks, nodata handling, and data range for every metric.
2. Fix integer overflow in SAM and validate RGB and multi-spectral behavior ([#50](https://github.com/nekhtiari/image-similarity-measures/issues/50)).
3. Resolve FSIM orientation assumptions and constant-image behavior ([#72](https://github.com/nekhtiari/image-similarity-measures/issues/72), [#62](https://github.com/nekhtiari/image-similarity-measures/issues/62)).
4. Independently validate the ISSM equation, histogram normalization, binning, and stabilizing constant before revisiting [#61](https://github.com/nekhtiari/image-similarity-measures/issues/61) and [#63](https://github.com/nekhtiari/image-similarity-measures/pull/63).
5. Make the 12-bit `max_p=4095` convention explicit and design a consistent `data_range` API ([#64](https://github.com/nekhtiari/image-similarity-measures/issues/64), [#28](https://github.com/nekhtiari/image-similarity-measures/issues/28)).
6. Specify mathematically meaningful results for identical, constant, zero, grayscale, and mismatched inputs.

Corrected behavior should first be opt-in or live behind a clearly versioned API. The project should publish side-by-side legacy and corrected fixture results before changing defaults.

## Next: useful extensions

Prioritize extensions that serve the existing remote-sensing users without forcing large dependencies on everyone:

- masked and nodata-aware evaluation;
- per-band scores alongside the existing aggregate scores;
- an explicit `data_range` option consistent across relevant metrics;
- structured input validation with actionable error messages;
- lazy loading of expensive optional FSIM dependencies;
- batch/file-pair evaluation and optional CSV or JSON Lines output;
- result provenance containing package version, loader, dtype, shape, and parameters;
- benchmark fixtures and performance work for [#29](https://github.com/nekhtiari/image-similarity-measures/issues/29);
- type checking and generated API documentation;
- optional modern perceptual metrics only in separate extras, so the base installation stays lightweight.

## Release and project operations

Before publishing 0.4.0:

1. review the local commits through a pull request;
2. verify GitHub repository ownership and PyPI project roles;
3. enable branch protection, private vulnerability reporting, and a protected release environment;
4. configure PyPI Trusted Publishing without a long-lived repository secret;
5. publish to TestPyPI or create a GitHub pre-release and test the exact artifacts;
6. publish to PyPI only after the compatibility comparison is approved.
