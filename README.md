# copernicus-matchup

Match in-situ observations against Copernicus Marine products and build aligned feature cubes.

Given a table of observations (latitude, longitude, time) and a list of gridded products, this
package downloads the products, cuts a space–time window around each observation, resamples the
windows onto a common grid, applies per-channel transformations, and writes one aligned cube per
feature group.

It knows nothing about what will be predicted from those cubes. The hand-off is on disk.

## Install

```bash
pip install "copernicus-matchup @ git+https://github.com/Kissyfur/copernicus-matchup@v0.1.0"
```

The download stage needs the Copernicus Marine client, which is an extra because its dependency
tree is large and `open_local_or_remote` works on local NetCDF without it:

```bash
pip install "copernicus-matchup[download] @ git+https://github.com/Kissyfur/copernicus-matchup@v0.1.0"
```

Add `[excel]` if your observation table is `.xls`/`.xlsx` rather than `.csv`.

## Use

Hand it a config and a run directory, and it builds the cubes:

```python
from copernicus_matchup import TargetTransform, build_dataset, load_data_config

config = load_data_config("data.yaml")
artifacts = build_dataset(
    config,
    run_root="outputs/my_run",
    target_transform=TargetTransform(transform="log", floor={"method": "positive_quantile", "q": 0.01}),
)
# {'target': ..., 'meta': ..., 'optics': ..., 'nut': ...}
```

`build_dataset` is the three stages in order. Run them separately when you want to
download once and re-matchup repeatedly, or matchup once and re-preprocess with different
transforms:

```python
from copernicus_matchup import create_matchups, download_products, preprocess_matchups

download_products(config, run_root)          # local products saved; remote ones recorded
create_matchups(config, run_root)            # target-centred windows per product
preprocess_matchups(config, run_root)        # aligned cubes per feature group
```

Each stage skips work that already exists unless given `overwrite=True`, so re-running is
cheap. `RunLayout` names every path they use:

```python
from copernicus_matchup import RunLayout

layout = RunLayout("outputs/my_run")
layout.raw, layout.targets, layout.matchups, layout.datasets
```

A minimal config:

```yaml
target:
  path: observations.csv
  target_column: concentration
  filter_column: SF            # optional row filter
  filter_values: ["0.2-3 µm"]
products:
  - name: reflectance
    dataset_ids: [cmems_obs-oc_med_bgc-reflectance_my_l3-multi-1km_P1D]
    variables: [RRS412, RRS443, RRS490]
    feature_group: optics
    preprocess:
      log: true
      positive_quantile: 0.01
      add_cloud_land_masks: true
      interpolate_dims: [lat, lon, time]
matchup:
  lat_window: 0.06
  lon_window: 0.06
  time_window_days: 14
  time_threshold_days: 1
preprocess:
  time_limit: 15
  time_selection: centered     # first | last | past_to_center | centered
  fillna: 0.0
regrid:
  enabled: true
  reference_group: optics      # coarser groups are resampled onto this one
  method: linear
```

Sections may also be wrapped in a `data:` block, so a project that adds its own modelling
sections alongside can keep one file.

## Command line

Installing the package puts `copernicus-matchup` on your PATH. It is the Python API with
a config path, for use without a surrounding project:

```bash
copernicus-matchup --config data.yaml --run-root outputs/my_run     --target-transform log --target-floor-quantile 0.01
```

It prints a JSON summary of what it wrote. `--stage` runs one stage at a time (repeatable),
which is how you re-matchup without re-downloading:

```bash
copernicus-matchup --config data.yaml --run-root outputs/my_run --stage download
copernicus-matchup --config data.yaml --run-root outputs/my_run --stage matchup --stage preprocess
```

## Output contract

One NetCDF `DataArray` per feature group, named after the group:

- dims `(Id, lat, lon, time, variable)` for cubes, `(Id, variable)` for `target` and `meta`
- `Id` indexes the observations and is the join key across groups
- `variable` names the channels, in the order the products were declared
- `lat`/`lon`/`time` are relative cube indices, not absolute coordinates — the real coordinates
  are consumed during the regrid and then dropped
- the target's quantile floor, when used, is recorded in the array's `attrs`

Read it back with `xarray.load_dataarray(path).sel(Id=ids)`.

Artifacts are always written as **NETCDF4**, stated explicitly rather than left to
xarray's engine search. Left to the default, an install without the netCDF4 backend falls
back to scipy and writes NetCDF3, which permits only one unlimited dimension and only at
index 0 -- so a metadata array with no columns (a zero-length second dimension) was
written in an illegal layout and could not be reopened at all. Pinning the format keeps
artifacts reproducible across environments.

## Transformation order

Applied per product, in this order, because several steps do not commute:

1. **Cloud and land masks** — derived from missingness, so this must run on the untouched array;
   any gap-filling beforehand destroys the signal being detected
2. **Positive-quantile flooring** — computed on observed values only, before gap-filling
3. **Gap filling** — linear interpolation along `interpolate_dims`
4. **Derived channels** — expressions over the product's own variables
5. **Monthly anomalies** — against the per-month climatology
6. **Log / log1p** — `log1p` for fields containing zeros
7. **Mask concatenation** — mask channels appended after the data channels
8. **Temporal selection** — `time_limit` steps, per `time_selection`
9. **Coverage filter** — `min_valid_ratio`; note that cloud cover is often correlated with the
   quantity being studied, so a strict threshold can bias the retained sample
10. **Residual imputation** — `fillna`, which lands *after* the log step, so an imputed value is
    zero in log space and should be read together with the mask channels

## Dependency policy

Floors only, no upper bounds. Output crosses the boundary as NetCDF and CSV, never as pickles or
`.npy`, so the writer's in-memory library versions do not enter the files. This was verified
rather than assumed: artifacts written under numpy 2.0.2 / pandas 2.3.3 / xarray 2024.7 were read
back bit-exactly under numpy 1.24.4 / pandas 2.1.4 / xarray 2023.11 — same values, dtypes, dims,
coords and attrs.

A consumer pinned to an older numpy (for example one held at `numpy<2` by TensorFlow 2.10) can
therefore read cubes produced by this package running on the latest numpy — **provided the two
live in separate environments**. In a shared environment pip still resolves one version per
package and the consumer's ceiling wins; the point of having no ceiling here is that this package
never contributes one.

## Tests

```bash
pip install -e ".[dev,download]"
pytest -q
```

## Status

Extracted from a Pseudo-nitzschia retrieval project, where it builds the environmental cubes.
The Python API and the CLI are both stable.
