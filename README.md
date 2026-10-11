# copernicus-matchup

Match in-situ observations against Copernicus Marine products and build aligned feature cubes.

You provide a table of observations (where, when, and the value measured) and a list of gridded
[Copernicus Marine](https://marine.copernicus.eu/) products such as ocean colour, temperature or
nutrients. For every observation, the package cuts a space–time window out of each product,
resamples the windows onto a common grid and applies per-channel transformations (log, gap
filling, cloud/land masks, anomalies and so on). It then writes one aligned NetCDF cube per
feature group, ready to train a model on.

The package knows nothing about what will be predicted from those cubes. The hand-off happens on
disk.

```
observations.csv ─┐
                  ├─► download ─► matchup ─► preprocess ─► datasets/{target, meta, <group>}.nc
data.yaml ────────┘
```

| Stage | What it does |
|---|---|
| `download` | Saves local products whole. For remote Copernicus products it only records how to open them: a whole dataset dwarfs the windows a run needs. |
| `matchup` | Standardises the observation table and cuts a window around every observation in every product. |
| `preprocess` | Transforms each product, stacks products into feature groups, regrids the groups onto one grid and writes the cubes. |

## Requirements

- **Python 3.12 or newer** for the pinned environment in `requirements.txt` (tested on 3.12).
- A free **[Copernicus Marine account](https://data.marine.copernicus.eu/register)** to read remote
  products. You don't need one if every product is a local NetCDF file.

## Installation

```bash
git clone https://github.com/Kissyfur/copernicus-matchup.git
cd copernicus-matchup

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-dev.txt    # requirements.txt alone if you don't need pytest
pip install -e .
```

Check that it works:

```bash
pytest -q
copernicus-matchup --help
```

Windows notes:
- With several Pythons installed, `py -3.12 -m venv .venv` picks the version.
- If PowerShell refuses to run `Activate.ps1`, allow local scripts once with
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Alternatively, skip activation and call
  `.venv\Scripts\python.exe` and `.venv\Scripts\copernicus-matchup.exe` directly.

`requirements.txt` pins every direct dependency to its latest tested release, including both
optional extras (the Copernicus Marine client and Excel support).

### As a dependency of another project

```bash
pip install "copernicus-matchup[download] @ git+https://github.com/Kissyfur/copernicus-matchup.git@v0.5.0"
```

This route resolves against the version floors in `pyproject.toml`, not the pins. The extras
are:

| Extra | What it adds |
|---|---|
| `download` | `copernicusmarine`, needed for remote products |
| `excel` | `openpyxl` and `xlrd`, for `.xlsx` and `.xls` observation tables |
| `dev` | `pytest` |

## Usage

### 1. Log in to Copernicus Marine (once)

```bash
copernicusmarine login
```

This stores your credentials under `~/.copernicusmarine`. Setting
`COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD` works too.

### 2. Prepare the observation table

Use a CSV, `.xlsx` or `.xls` file with one row per observation:

| Id | lat | lon | time | concentration |
|---|---|---|---|---|
| 1 | 41.38 | 2.19 | 2021-03-14 | 1250.0 |
| 2 | 41°22'48"N | 2°11'24"E | 2021-03-15 | 0.0 |

- Column names are configurable. An `Id` column is generated if there is none.
- Coordinates may be decimal degrees or degree-minute-second text.
- Rows missing a coordinate, the time or the target are dropped.

### 3. Write a data config

The `data.yaml` below pairs a gappy L3 ocean-colour product, which supplies the cloud mask,
with a gap-free L4 nutrient model, which supplies the land mask. It then resamples the coarser
nutrient grid onto the ocean-colour one:

```yaml
target:
  path: observations.csv
  target_column: concentration        # a list gives several outputs, kept in that order

products:
  - name: reflectance
    dataset_ids: [cmems_obs-oc_med_bgc-reflectance_my_l3-multi-1km_P1D]
    variables: [RRS412, RRS443, RRS490]
    feature_group: optics
    preprocess:
      log: true
      add_cloud_land_masks: true
      mask_kinds: [cloud_mask]
      interpolate_dims: [lat, lon, time]

  - name: nutrients
    dataset_ids: [cmems_mod_med_bgc-nut_anfc_4.2km_P1D-m]
    variables: [nh4, no3, po4]
    feature_group: nut
    open_dataset_kwargs: {minimum_depth: 0, maximum_depth: 2}   # surface only
    preprocess:
      log: false
      log1p: true
      add_cloud_land_masks: true
      mask_kinds: [land_mask]
      interpolate_dims: [lat, lon, time]

matchup:
  lat_window: 0.06              # half-width of the window, degrees
  lon_window: 0.06
  time_window_days: 14          # half-width of the window, days
  time_threshold_days: 1

preprocess:                     # defaults for every product
  time_limit: 15
  time_selection: centered      # first | last | past_to_center | centered
  fillna: 0.0

regrid:
  enabled: true
  reference_group: optics       # other groups are resampled onto this grid
  method: linear
```

To use a NetCDF file you already have instead of a remote product, give the product
`source: local` and `source_path: path/to/file.nc`. Relative paths in the config are resolved
against the directory you run from, not the config file's directory. The sections may also be
wrapped in a `data:` block, so a project with its own modelling sections can keep a single file.
JSON works as well as YAML.

### 4. Run it

From the command line:

```bash
copernicus-matchup --config data.yaml --run-root outputs/my_run --target-transform log --target-floor-quantile 0.01
```

The command prints a JSON summary of the files it wrote. `--stage` runs a single stage and can
be repeated. That is how you re-run the matchup without downloading again:

```bash
copernicus-matchup --config data.yaml --run-root outputs/my_run --stage download
copernicus-matchup --config data.yaml --run-root outputs/my_run --stage matchup --stage preprocess
```

| Option | Meaning |
|---|---|
| `--config` | Data config, YAML or JSON (required) |
| `--run-root` | Output directory (required) |
| `--stage` | `download`, `matchup` or `preprocess`; repeatable; default is all three |
| `--overwrite` | Redo `download` and `matchup` outputs that already exist (`preprocess` always rebuilds) |
| `--target-transform` | `none` (default), `log` or `log1p` |
| `--target-offset` | Added to the target before the transform |
| `--target-floor-quantile` | Raise targets up to this quantile of the positive values first; required for `log` when the target has zeros |
| `--quiet` | Only report warnings and errors |

From Python, the equivalent is:

```python
from copernicus_matchup import TargetTransform, build_dataset, load_data_config

config = load_data_config("data.yaml")
artifacts = build_dataset(
    config,
    run_root="outputs/my_run",
    target_transform=TargetTransform(transform="log", floor={"method": "positive_quantile", "q": 0.01}),
)
# {'target': Path(...), 'meta': Path(...), 'optics': Path(...), 'nut': Path(...)}
```

`build_dataset` runs `download_products`, `create_matchups` and `preprocess_matchups` in order.
Each of the three can also be called on its own with the same `(config, run_root)` arguments.

### 5. Read the results

```
outputs/my_run/
├── raw/                          # local products saved whole, <name>.remote.json for remote ones
├── processed/
│   ├── targets.csv               # standardised observation table
│   └── matchups/
│       ├── <product>.nc          # one window per matched observation
│       └── <product>_unmatched.csv
└── datasets/
    ├── target.nc                 # (Id, variable)
    ├── meta.nc                   # (Id, variable): lat, lon, x_day, y_day, day, metadata_columns
    └── <feature_group>.nc        # (Id, lat, lon, time, variable)
```

```python
import numpy as np
import xarray as xr

optics = xr.load_dataarray("outputs/my_run/datasets/optics.nc")
target = xr.load_dataarray("outputs/my_run/datasets/target.nc")
ids = np.intersect1d(optics["Id"], target["Id"])     # observations every product matched
X, y = optics.sel(Id=ids), target.sel(Id=ids)
```

- `Id` indexes the observations and is the join key across files. `target.nc` holds every
  observation, while each cube holds only those its products matched.
- `variable` names the channels, in the order the products and variables were declared. Mask
  channels come after the data channels.
- `lat`, `lon` and `time` in the cubes are relative indices, not coordinates. The real
  coordinates are used for regridding and then dropped.
- A single target is stored as variable `target`. A list of targets keeps the column names.
- The target's quantile floor, when used, is recorded in the array's `attrs`.
- Files are always written as NETCDF4 (HDF5-based), never NetCDF3, so they read back the same in
  every environment.

## Configuration reference

**`target`**

| Key | Default | Meaning |
|---|---|---|
| `path` | required | Observation table: `.csv`, `.xlsx` or `.xls` |
| `target_column` | required | Column to predict, or a list of columns |
| `id_column`, `lat_column`, `lon_column`, `time_column` | `Id`, `lat`, `lon`, `time` | Column names in your table |
| `sheet_name` | first sheet | Excel sheet name or index |
| `filter_column`, `filter_values` | — | Keep only rows whose `filter_column` value is listed |
| `metadata_columns` | `[]` | Extra columns copied into `meta.nc` |
| `include_spatial_metadata` | `true` | Add `lat`, `lon` to `meta.nc` |
| `include_day_metadata` | `true` | Add day of year |
| `include_cyclic_day_metadata` | `true` | Add the day of year as a point on a circle (`x_day`, `y_day`) |

**`products`** (a list)

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Used in file names, and as the feature group if none is given |
| `dataset_ids` | required for remote | Copernicus dataset IDs, tried in order until one opens |
| `variables` | all | Variables to keep, in channel order |
| `feature_group` | `name` | Products sharing a group are stacked into one cube |
| `source`, `source_path` | `copernicus` | `source: local` reads `source_path` instead |
| `open_dataset_kwargs` | `{}` | Passed to `copernicusmarine.open_dataset`, e.g. `minimum_depth`, `start_datetime` or a bounding box |
| `rename_dimensions` | `latitude→lat`, `longitude→lon` | Maps the product's dimension names onto `lat`/`lon`/`time` |
| `rename_variables` | `{}` | Renames variables |
| `matchup` | `{}` | Per-product overrides of the `matchup` section |
| `preprocess` | `{}` | Per-product options, see below |

**`matchup`**

| Key | Default | Meaning |
|---|---|---|
| `lat_window`, `lon_window` | `0.06` | Half-width of the window around the nearest grid cell, degrees |
| `time_window_days` | `1` | Half-width of the window, days |
| `lat_threshold`, `lon_threshold` | `0.1` | An observation is unmatched if the nearest grid cell is at least this far away, degrees |
| `time_threshold_days` | `1` | The same, for the nearest time step, in days |
| `require_full_time_window` | `false` | Drop windows with fewer than `2 × time_window_days + 1` time steps |

Windows clipped by the edge of a product's domain are recorded as unmatched too.

**`preprocess`**: the top-level section sets defaults for every product, and
`products[].preprocess` overrides them. Keys marked *product* are read only at product level.

| Key | Default | Meaning |
|---|---|---|
| `log` | `true` | Natural log of the data channels (named `log_products` in the top-level section) |
| `log1p` *product* | `false` | `log(1 + x)`, for fields that contain zeros |
| `exclude_from_log`, `exclude_from_log1p` *product* | `[]` | Channels to leave untransformed |
| `positive_quantile` | `0.01` | Raise values below this quantile of the positive values up to it; `null` disables it |
| `interpolate_dims` *product* | `[]` | Fill gaps by linear interpolation along these dimensions |
| `add_cloud_land_masks` | `true` | Append mask channels derived from missing values |
| `mask_kinds` *product* | both | `[cloud_mask]` for an L3 product, `[land_mask]` for a gap-free L4 product |
| `add_photoperiod` *product* | `false` | Append a daylight-hours channel |
| `derived_variables` *product* | `[]` | `{name, expression}` channels built from the product's variables with `+ - * / **` |
| `anomaly_variables` *product* | `[]` | Append `<name>_anom`, the anomaly against the monthly climatology |
| `time_limit` | all | Number of time steps to keep |
| `time_selection` | `first` | Which steps to keep: `first`, `last`, `past_to_center` or `centered` |
| `min_valid_ratio` | — | Drop observations whose window has a smaller share of cloud-free water than this (needs a cloud mask) |
| `fillna` | `0.0` | Value for the gaps that remain; `null` keeps NaN |
| `prefix_variables` | `false` | Name channels `<product>:<variable>` |

The defaults suit positive concentrations such as reflectance, chlorophyll or nutrients. For a
field that can be negative or shouldn't be logged, such as temperature or currents, set
`log: false` and `positive_quantile: null` on that product.

**`regrid`**

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Resample every feature group onto the reference group's lat/lon grid |
| `reference_group` | — | Usually the finest-resolution group |
| `method` | `linear` | `linear` or `nearest` |

## Transformation order

The steps run per product, in this order, because several of them do not commute:

1. **Cloud and land masks**: derived from missing values, so they must see the untouched array
2. **Positive-quantile floor**: computed on observed values only, before gap filling
3. **Gap filling** along `interpolate_dims`
4. **Derived channels**, then **monthly anomalies**
5. **Log / log1p**
6. **Mask and photoperiod channels** appended after the data channels
7. **Temporal selection** (`time_limit`, `time_selection`)
8. **Coverage filter** (`min_valid_ratio`). Cloud cover often correlates with the quantity being
   studied, so a strict threshold can bias the sample that remains.
9. **Residual imputation** (`fillna`). This runs after the log, so an imputed value is zero in log
   space and should be read together with the mask channels.

## Dependency policy

`pyproject.toml` declares version floors only, never ceilings. Output crosses the boundary as
NetCDF and CSV, never as pickles or `.npy`, so the writer's library versions don't enter the
files. Artifacts written under numpy 2.0.2, pandas 2.3.3 and xarray 2024.7 read back bit-exactly
under numpy 1.24.4, pandas 2.1.4 and xarray 2023.11.

A consumer held at an older numpy, such as TensorFlow 2.10 at `numpy<2`, can therefore read
cubes produced here on the latest stack, as long as the two live in separate environments.
`requirements.txt` is the other half: the exact, latest set this package was tested with.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Status

Extracted from a *Pseudo-nitzschia* retrieval project, where it builds the environmental cubes.
The Python API and the CLI are both stable.
