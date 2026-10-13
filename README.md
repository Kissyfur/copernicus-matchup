# copernicus-matchup

Match in-situ observations against [Copernicus Marine](https://marine.copernicus.eu/) products
and build aligned feature cubes for machine learning.

You give it a table of observations (where, when, value) and a list of gridded products such as
ocean colour, temperature or nutrients. For each observation it cuts a space–time window out of
every product and transforms the channels (log, gap filling, cloud/land masks, anomalies). It
then resamples everything onto one grid and writes one NetCDF cube per feature group. What you
predict from the cubes is up to you.

```
observations.csv + data.yaml ─► download ─► matchup ─► preprocess ─► datasets/*.nc
```

| Stage | What it does |
|---|---|
| `download` | Saves local products. Remote products are only recorded, because the matchup reads just the windows it needs |
| `matchup` | Cuts a window around every observation in every product |
| `preprocess` | Transforms, groups and regrids the windows, then writes the cubes |

## Installation

You need Python 3.12+ for the pinned environment, and a free
[Copernicus Marine account](https://data.marine.copernicus.eu/register) for remote products.

```bash
git clone https://github.com/Kissyfur/copernicus-matchup.git
cd copernicus-matchup
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell
source .venv/bin/activate             # Linux / macOS
pip install -r requirements-dev.txt   # latest tested versions (requirements.txt skips pytest)
pip install -e .
pytest -q                             # check that everything works
```

On Windows, `py -3.12 -m venv .venv` picks the Python version. If PowerShell blocks
`Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### Inside another project

The package installs on Python 3.9+ and adapts to the project's environment.
`pyproject.toml` only sets minimum versions, all of them tested, so pip keeps any library you
already have that meets them.

```bash
pip install "copernicus-matchup[download] @ git+https://github.com/Kissyfur/copernicus-matchup.git@v0.7.0"
```

- **Minimum versions:** numpy 1.22.4, pandas 2.1, xarray 2023.11, scipy 1.10, netCDF4 1.6 and
  PyYAML 6.0.
- **Extras:** `download` (Copernicus Marine client, for remote products), `excel` (`.xlsx`/`.xls`
  tables) and `dev` (pytest).
- Add `--dry-run` to the install command to see what pip would change.

**TensorFlow 2.10 (GPU on native Windows).** Add these lines to the project's requirements:

```text
tensorflow==2.10.1
numpy<2          # TensorFlow 2.10 needs numpy 1.x but doesn't declare it
setuptools<82    # TensorBoard 2.10 needs pkg_resources; the Copernicus client upgrades setuptools
copernicus-matchup[download] @ git+https://github.com/Kissyfur/copernicus-matchup.git@v0.7.0
```

This setup is tested with Python 3.9 on an NVIDIA GPU, with TensorFlow and the cubes in one
process. On Python < 3.10, the Copernicus client also downgrades urllib3 to 1.26, which doesn't
affect TensorFlow.

## Usage

### 1. Log in to Copernicus Marine (once)

```bash
copernicusmarine login
```

### 2. Prepare the observations

Use a CSV, `.xlsx` or `.xls` table with one row per observation:

| Id | lat | lon | time | concentration |
|---|---|---|---|---|
| 1 | 41.38 | 2.19 | 2021-03-14 | 1250.0 |
| 2 | 41°22'48"N | 2°11'24"E | 2021-03-15 | 0.0 |

- Column names are configurable, and `Id` is generated if missing.
- Coordinates may be decimal or degree-minute-second text.
- Incomplete rows are dropped.

### 3. Write a config

This `data.yaml` pairs a gappy L3 ocean-colour product, which supplies the cloud mask, with a
gap-free L4 nutrient model, which supplies the land mask. The nutrients are then resampled onto
the finer ocean-colour grid:

```yaml
target:
  path: observations.csv
  target_column: concentration        # a list gives several outputs

products:
  - name: reflectance
    dataset_ids: [cmems_obs-oc_med_bgc-reflectance_my_l3-multi-1km_P1D]
    variables: [RRS412, RRS443, RRS490]
    feature_group: optics
    preprocess:
      log: true
      mask_kinds: [cloud_mask]
      interpolate_dims: [lat, lon, time]

  - name: nutrients
    dataset_ids: [cmems_mod_med_bgc-nut_anfc_4.2km_P1D-m]
    variables: [nh4, no3, po4]
    feature_group: nut
    open_dataset_kwargs: {minimum_depth: 0, maximum_depth: 2}   # surface only
    matchup: {lat_window: 0.12, lon_window: 0.12}   # coarse grid: cover the optics window
    preprocess:
      log: false
      log1p: true
      mask_kinds: [land_mask]
      interpolate_dims: [lat, lon, time]

matchup:
  lat_window: 0.06              # half-width, degrees
  lon_window: 0.06
  time_window_days: 14          # half-width, days

preprocess:                     # defaults for every product
  time_limit: 15
  time_selection: centered

regrid:
  enabled: true
  reference_group: optics       # other groups are resampled onto this grid
```

- **Local files:** for a NetCDF file you already have, use `source: local` and
  `source_path: file.nc` instead of `dataset_ids`.
- **Relative paths** are resolved from the directory you run in.
- **Format:** the sections may sit under a `data:` key, to share a file with other settings.
  JSON works too.

### 4. Run

```bash
copernicus-matchup --config data.yaml --run-root outputs/my_run --target-transform log --target-floor-quantile 0.01
```

- `--target-floor-quantile` raises the target values below that quantile of the positive values
  up to it, so zeros survive the log.
- `--stage download|matchup|preprocess` runs single stages and can be repeated, for example to
  re-run the matchup without downloading again.
- `--overwrite` redoes outputs that already exist.
- `copernicus-matchup --help` lists every option.

The same from Python:

```python
from copernicus_matchup import TargetTransform, build_dataset, load_data_config

artifacts = build_dataset(
    load_data_config("data.yaml"),
    run_root="outputs/my_run",
    target_transform=TargetTransform(transform="log", floor={"method": "positive_quantile", "q": 0.01}),
)
```

`build_dataset` runs `download_products`, `create_matchups` and `preprocess_matchups` in order.
Each can also be called on its own.

### 5. Read the results

```
outputs/my_run/
├── raw/                    # local products, markers for remote ones
├── processed/              # targets.csv, per-product windows, unmatched observations
└── datasets/
    ├── target.nc           # (Id, variable)
    ├── meta.nc             # (Id, variable): lat, lon, day of year, metadata_columns
    └── <feature_group>.nc  # (Id, lat, lon, time, variable)
```

```python
import numpy as np
import xarray as xr

optics = xr.load_dataarray("outputs/my_run/datasets/optics.nc")
target = xr.load_dataarray("outputs/my_run/datasets/target.nc")
ids = np.intersect1d(optics["Id"], target["Id"])   # target.nc also keeps unmatched observations
X, y = optics.sel(Id=ids), target.sel(Id=ids)
```

- `Id` is the join key across files.
- `variable` lists the channels in the order they were declared, with masks last.
- `lat`, `lon` and `time` are relative indices, not coordinates.
- A single target is named `target`. A list of targets keeps its column names.
- The files are NetCDF4, so environments on older numpy and xarray can read them too.

## Configuration reference

**`target`**

| Key | Default | Meaning |
|---|---|---|
| `path`, `target_column` | required | Table file; the column to predict, or a list of columns |
| `id_column`, `lat_column`, `lon_column`, `time_column` | `Id`, `lat`, `lon`, `time` | Column names in the table |
| `sheet_name` | first sheet | Excel sheet name or index |
| `filter_column`, `filter_values` | — | Keep only rows whose value is listed |
| `keep_missing_targets` | `false` | Keep rows with an empty target, to build cubes for prediction |
| `metadata_columns` | `[]` | Extra columns copied into `meta.nc` |
| `include_spatial_metadata`, `include_day_metadata`, `include_cyclic_day_metadata` | `true` | Add lat/lon, day of year, and its cosine and sine (`x_day`, `y_day`) to `meta.nc` |

**`products`** (a list)

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Used in file names, and as the feature group if none is given |
| `dataset_ids` | — | Copernicus dataset IDs, tried in order until one opens |
| `source`, `source_path` | `copernicus` | `source: local` reads `source_path` instead |
| `variables` | all | Variables to keep, in channel order |
| `feature_group` | `name` | Products in the same group share a cube |
| `open_dataset_kwargs` | `{}` | Passed to `copernicusmarine.open_dataset` (depth, dates, bounding box) |
| `rename_dimensions`, `rename_variables` | `latitude→lat`, `longitude→lon` | Map the product's names onto `lat`/`lon`/`time` |
| `matchup`, `preprocess` | `{}` | Per-product overrides of those sections |

**`matchup`**

| Key | Default | Meaning |
|---|---|---|
| `lat_window`, `lon_window` | `0.06` | Window half-width around the nearest grid cell, in degrees |
| `time_window_days` | `1` | Window half-width, in days |
| `lat_threshold`, `lon_threshold` | `0.1` | Unmatched if the nearest grid cell is at least this far away, in degrees |
| `time_threshold_days` | `1` | Unmatched if the nearest time step is at least this far away, in days |
| `require_full_time_window` | `false` | Drop windows with missing time steps |

Observations at the edge of a product's domain get clipped windows, and are recorded as
unmatched.

**`preprocess`**: the top-level section sets defaults and `products[].preprocess` overrides
them. Keys marked *product* only work at product level.

| Key | Default | Meaning |
|---|---|---|
| `log` | `true` | Natural log of the data channels (`log_products` at top level) |
| `log1p` *product* | `false` | `log(1 + x)`, for fields with zeros |
| `exclude_from_log`, `exclude_from_log1p` *product* | `[]` | Channels left untransformed |
| `positive_quantile` | `0.01` | Raise values below this quantile of the positive values up to it; `null` disables it |
| `interpolate_dims` *product* | `[]` | Fill gaps by linear interpolation along these dimensions |
| `add_cloud_land_masks` | `true` | Append mask channels derived from missing values |
| `mask_kinds` *product* | both | `[cloud_mask]` for an L3 product, `[land_mask]` for a gap-free L4 product |
| `add_photoperiod` *product* | `false` | Append a daylight-hours channel |
| `derived_variables` *product* | `[]` | `{name, expression}` channels computed with `+ - * / **` |
| `anomaly_variables` *product* | `[]` | Append `<name>_anom`, the anomaly against the monthly climatology |
| `time_limit` | all | Number of time steps to keep |
| `time_selection` | `first` | Which steps: `first`, `last`, `past_to_center` or `centered` |
| `min_valid_ratio` | — | Drop observations with a smaller share of cloud-free water (needs a cloud mask) |
| `fillna` | `0.0` | Value for the remaining gaps; `null` keeps NaN |
| `prefix_variables` | `false` | Name channels `<product>:<variable>` |

The defaults suit positive concentrations. For temperature, currents or any field that can be
negative, set `log: false` and `positive_quantile: null` on that product.

**`regrid`**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Resample every feature group onto the reference group's grid |
| `reference_group` | — | Usually the finest-resolution group |
| `method` | `linear` | `linear` or `nearest` |

Regridding can't extrapolate. Give coarser products a wider `matchup` window than the reference
group, as in the example. Otherwise, the cells outside their coverage stay NaN.

## Transformation order

Each product goes through these steps in order, because several of them do not commute:

1. Cloud and land masks, computed on the untouched data
2. Positive-quantile floor, on observed values only
3. Gap filling
4. Derived channels, then anomalies
5. Log / log1p
6. Mask and photoperiod channels appended
7. Time selection
8. Coverage filter (`min_valid_ratio`). Cloud cover often correlates with the quantity being
   measured, so a strict threshold can bias the sample.
9. `fillna`. This runs after the log, so filled values are 0 in log space. Read them together
   with the mask channels.

---

Extracted from a *Pseudo-nitzschia* retrieval project, where it builds the environmental cubes.
The Python API and the CLI are stable.
