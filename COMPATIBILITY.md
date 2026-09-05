# Compatibility policy

Image Similarity Measures treats its import paths, function signatures, command-line interface, loading behavior, and numerical results as compatibility surfaces.

## Versioning

The project follows semantic versioning and uses a deprecation-first policy:

- patch releases fix defects without intentionally changing supported outputs or public APIs;
- minor releases may add functionality, dependencies, or opt-in corrected behavior;
- breaking public API or default numerical changes require advance notice, migration guidance, and a major release unless needed for security.

Python and dependency support changes are documented in the changelog. Python 3.10 is retained for the 0.4 series and reaches upstream end of life in October 2026; its eventual removal will be announced separately.

## Public API in the 0.4 series

The compatibility suite protects these existing entry points:

- `image_similarity_measures.evaluate.evaluation`
- `image_similarity_measures.evaluate.read_image`
- `image_similarity_measures.quality_metrics.{fsim,issm,psnr,rmse,sam,sre,ssim,uiq}`
- `image_similarity_measures.quality_metrics.metric_functions`
- the `image-similarity-measures` command and its JSON result structure

Internal helpers with leading underscores are not public APIs.

## Numerical changes

Floating-point results can vary slightly across supported platforms and dependency versions. Intentional formula, dtype, scaling, or zero-division changes must be isolated, documented, and validated against a primary reference or an independent implementation.

Version 0.4 is a compatibility and packaging release. It deliberately preserves the 0.3.6 metric implementations, including known edge cases, so that modernization does not silently alter downstream results.

## Image loading

`read_image(path)` preserves the historical automatic loader selection:

1. if Rasterio is importable and `path` ends in lowercase `.tif` or `.tiff`, Rasterio is used and its band-first result is converted to channel-last;
2. otherwise OpenCV is used.

The loaders may return different band counts, dtypes, and value ranges for the same TIFF. Installing the `rasterio` extra can therefore change numerical results. Explicit loader selection is planned as an additive API; the legacy automatic default will remain available.

## Documented legacy limitations

The following behaviors are retained in 0.4 and covered by characterization tests:

- two-dimensional grayscale input is handled inconsistently across metrics;
- SAM can overflow when multiplying integer arrays;
- some metrics produce `NaN`, infinity, or unintuitive values for constant or identical inputs;
- ISSM has unresolved formula and stability questions;
- shape validation currently raises `AssertionError` and can be disabled by optimized Python;
- RMSE, PSNR, and SSIM default to the original 12-bit `max_p=4095` convention;
- TIFF loader selection depends on the installed optional dependencies and a case-sensitive suffix.

These limitations will be addressed as separately reviewed changes after the compatibility release. Applications that depend on exact results should pin the package and its dependency versions.
