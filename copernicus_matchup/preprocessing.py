from __future__ import annotations

import ast
import logging
import operator
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from copernicus_matchup.config import DataConfig, ProductSpec, RegridConfig
from copernicus_matchup.targets import load_target_table, metadata_to_dataarray, target_to_dataarray
from copernicus_matchup.features.astronomy import photoperiod_hours
from copernicus_matchup.features.masks import get_cloud_and_land_masks, valid_water_coverage
from copernicus_matchup.features.regrid import resample_group_to_reference
from copernicus_matchup.features.transforms import interpolate_gaps, monthly_anomaly, positive_quantile
from copernicus_matchup._utils import as_list

logger = logging.getLogger("copernicus_matchup.preprocess")

VARIABLE = "variable"
ORDERED_CUBE_DIMS = ("Id", "lat", "lon", "time", VARIABLE)
_DERIVED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: "safe_divide",
    ast.Pow: operator.pow,
}
_DERIVED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _option(product: ProductSpec, key: str, default: Any) -> Any:
    return product.preprocess.get(key, default)


def _select_time(data: xr.DataArray, limit: int | None, selection: str = "first") -> xr.DataArray:
    if limit is None or "time" not in data.dims:
        return data
    count = data.sizes["time"]
    limit = min(int(limit), count)
    selection = selection.lower().replace("-", "_")
    if selection in {"first", "start"}:
        return data.isel(time=range(limit))
    if selection in {"last", "end"}:
        return data.isel(time=range(count - limit, count))
    if selection in {"past_to_center", "center_past", "up_to_center"}:
        end = count // 2 + 1
        start = max(0, end - limit)
        return data.isel(time=range(start, end))
    if selection in {"centered", "center", "symmetric"}:
        center = count // 2
        start = max(0, center - limit // 2)
        end = min(count, start + limit)
        start = max(0, end - limit)
        return data.isel(time=range(start, end))
    raise ValueError(
        "preprocess.time_selection must be 'first', 'last', 'past_to_center', or 'centered'. "
        f"Got: {selection}"
    )


def _use_relative_cube_coordinates(data: xr.DataArray) -> xr.DataArray:
    coords = {
        dim: np.arange(data.sizes[dim], dtype=np.int32)
        for dim in ("lat", "lon", "time")
        if dim in data.dims
    }
    return data.assign_coords(coords) if coords else data


def _ordered_common_ids(arrays: list[xr.DataArray], group_name: str) -> np.ndarray:
    if not arrays or any("Id" not in array.dims for array in arrays):
        return np.array([])
    first_ids = arrays[0]["Id"].values
    other_id_sets = [set(array["Id"].values.tolist()) for array in arrays[1:]]
    common_ids = [id_ for id_ in first_ids if all(id_ in ids for ids in other_id_sets)]
    if not common_ids:
        raise ValueError(
            f"No common Id values remain after preprocessing products in feature group '{group_name}'. "
            "Check unmatched observations and product-level valid-data filters."
        )
    return np.asarray(common_ids, dtype=first_ids.dtype)


def _align_to_common_ids(arrays: list[xr.DataArray], group_name: str) -> list[xr.DataArray]:
    if len(arrays) <= 1 or any("Id" not in array.dims for array in arrays):
        return arrays
    common_ids = _ordered_common_ids(arrays, group_name)
    return [array.sel(Id=common_ids) for array in arrays]


def _as_dataarray(ds: xr.Dataset, product: ProductSpec) -> xr.DataArray:
    variables = product.variables or list(ds.data_vars)
    variables = [product.rename_variables.get(var, var) for var in variables]
    ds = ds[variables]
    if product.rename_variables:
        rename_map = {old: new for old, new in product.rename_variables.items() if old in ds.data_vars}
        ds = ds.rename(rename_map)
    return ds.to_array(dim=VARIABLE)


def _safe_log(data: xr.DataArray) -> xr.DataArray:
    return np.log(data.where(data > 0))


def _safe_log1p(data: xr.DataArray) -> xr.DataArray:
    return np.log1p(data.where(data > -1))


def _transform_except_variables(
    data: xr.DataArray,
    transform,
    excluded_variables: list[Any],
) -> xr.DataArray:
    excluded = {str(value) for value in excluded_variables}
    if not excluded or VARIABLE not in data.coords:
        return transform(data)

    transformed = transform(data)
    keep_original = xr.DataArray(
        [str(value) in excluded for value in data[VARIABLE].values],
        dims=[VARIABLE],
        coords={VARIABLE: data[VARIABLE].values},
    )
    return data.where(keep_original, transformed)


def _safe_divide(left: xr.DataArray, right: xr.DataArray, epsilon: float) -> xr.DataArray:
    return left / right.where(np.abs(right) > epsilon)


def _evaluate_derived_expression(
    node: ast.AST,
    variables: dict[str, xr.DataArray],
    epsilon: float,
) -> xr.DataArray | float:
    if isinstance(node, ast.Expression):
        return _evaluate_derived_expression(node.body, variables, epsilon)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Derived variable expression references unknown variable: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left = _evaluate_derived_expression(node.left, variables, epsilon)
        right = _evaluate_derived_expression(node.right, variables, epsilon)
        op = _DERIVED_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator in derived variable expression: {type(node.op).__name__}")
        if op == "safe_divide":
            return _safe_divide(left, right, epsilon)
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_derived_expression(node.operand, variables, epsilon)
        op = _DERIVED_UNARY_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator in derived variable expression: {type(node.op).__name__}")
        return op(operand)
    raise ValueError(f"Unsupported syntax in derived variable expression: {type(node).__name__}")


def _derived_variable_specs(product: ProductSpec) -> list[dict[str, str]]:
    specs = as_list(_option(product, "derived_variables", None))
    normalized = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each derived variable must be a mapping with 'name' and 'expression'.")
        name = str(spec.get("name", "")).strip()
        expression = str(spec.get("expression", spec.get("expr", ""))).strip()
        if not name or not expression:
            raise ValueError("Each derived variable must define non-empty 'name' and 'expression'.")
        normalized.append({"name": name, "expression": expression})
    return normalized


def _add_derived_variables(data: xr.DataArray, product: ProductSpec) -> xr.DataArray:
    specs = _derived_variable_specs(product)
    if not specs:
        return data
    if VARIABLE not in data.coords:
        raise ValueError("Derived variables require a 'variable' coordinate.")

    epsilon = float(_option(product, "derived_variables_epsilon", 1e-12))
    variables = {str(name): data.sel({VARIABLE: name}) for name in data[VARIABLE].values}
    derived = []
    for spec in specs:
        tree = ast.parse(spec["expression"], mode="eval")
        value = _evaluate_derived_expression(tree, variables, epsilon)
        if not isinstance(value, xr.DataArray):
            value = xr.zeros_like(next(iter(variables.values()))) + float(value)
        value = value.expand_dims({VARIABLE: [spec["name"]]})
        derived.append(value)
        variables[spec["name"]] = value.sel({VARIABLE: spec["name"]})
    return xr.concat([data] + derived, dim=VARIABLE, coords="minimal")


def _add_anomaly_variables(data: xr.DataArray, product: ProductSpec) -> xr.DataArray:
    names = as_list(_option(product, "anomaly_variables", None))
    if not names:
        return data
    if VARIABLE not in data.coords:
        raise ValueError("Anomaly variables require a 'variable' coordinate.")

    available = set(data[VARIABLE].values.tolist())
    anomalies = []
    for name in names:
        name = str(name)
        if name not in available:
            raise ValueError(f"Anomaly variable references unknown variable: {name}")
        anomaly = monthly_anomaly(data.sel({VARIABLE: name})).expand_dims({VARIABLE: [f"{name}_anom"]})
        anomalies.append(anomaly)
    return xr.concat([data] + anomalies, dim=VARIABLE, coords="minimal")


def _photoperiod_channel(data: xr.DataArray) -> xr.DataArray:
    """Hours of daylight per (Id, lat-pixel, time-step), from each pixel's own real
    latitude and each time-step's own real date -- not from the target observation's
    single (lat, date), so the channel carries genuine day-to-day variation across the
    matchup window instead of one value repeated across every time-step. Independent of
    longitude, so broadcast_like expands it across lon by plain repetition, not
    interpolation.
    """
    day_of_year = data["time"].dt.dayofyear
    photoperiod = photoperiod_hours(data["lat"], day_of_year)
    photoperiod = photoperiod.broadcast_like(data.isel({VARIABLE: 0}, drop=True))
    return photoperiod.astype(np.float32).expand_dims({VARIABLE: ["photoperiod"]})


def _prepare_product_array(
    ds: xr.Dataset,
    product: ProductSpec,
    defaults: DataConfig,
) -> xr.DataArray:
    """Returns real per-sample lat/lon/time coordinates (from `_match_one`), not yet reset
    to relative indices -- `preprocess_matchups` resets them once, after the optional
    regrid step, since regridding needs the real coordinates to interpolate against.
    """
    data = _as_dataarray(ds, product)

    # Cloud/land detection FIRST, on the untouched array. get_cloud_and_land_masks reads
    # clouds straight off np.isnan, so any gap-filling done before it erases the very
    # thing it is looking for -- interpolating first left cloud_mask flagging ~3% of
    # pixels in a daily Mediterranean L3 scene, and filled interior land from the water
    # around it.
    add_masks = bool(_option(product, "add_cloud_land_masks", defaults.preprocess.add_cloud_land_masks))
    cloud_mask = land_mask = None
    if add_masks:
        cloud_mask, land_mask = get_cloud_and_land_masks(data, variable_dim=VARIABLE)

    photoperiod = None
    if bool(_option(product, "add_photoperiod", False)):
        photoperiod = _photoperiod_channel(data)

    # Then the positive-quantile floor, still on real observations only: a quantile taken
    # over interpolated values is a quantile of partly invented data.
    quantile = _option(product, "positive_quantile", defaults.preprocess.positive_quantile)
    if quantile is not None:
        quantile_dims = tuple(
            _option(product, "positive_quantile_dims", ("Id", "time", "lat", "lon"))
        )
        data = positive_quantile(data, quantile=float(quantile), dims=quantile_dims)

    # Only now fill the gaps. Derived variables and anomalies are computed after this so
    # they are not riddled with holes propagated from their inputs; moving this below the
    # log/log1p block instead would interpolate in log space, which is a one-line change
    # if that turns out to suit the optics channels better.
    interpolate_dims = tuple(_option(product, "interpolate_dims", ()))
    if interpolate_dims:
        data = interpolate_gaps(data, interpolate_dims)

    data = _add_derived_variables(data, product)
    data = _add_anomaly_variables(data, product)

    if bool(_option(product, "log", defaults.preprocess.log_products)):
        data = _transform_except_variables(data, _safe_log, as_list(_option(product, "exclude_from_log", None)))
    if bool(_option(product, "log1p", False)):
        data = _transform_except_variables(data, _safe_log1p, as_list(_option(product, "exclude_from_log1p", None)))

    if bool(_option(product, "prefix_variables", defaults.preprocess.prefix_variables)):
        names = [f"{product.name}:{name}" for name in data[VARIABLE].values]
        data = data.assign_coords({VARIABLE: names})

    arrays = [data]
    if cloud_mask is not None and land_mask is not None:
        mask_kinds = {str(value) for value in as_list(_option(product, "mask_kinds", ["cloud_mask", "land_mask"]))}
        if "cloud_mask" in mask_kinds:
            arrays.append(cloud_mask)
        if "land_mask" in mask_kinds:
            arrays.append(land_mask)
    if photoperiod is not None:
        arrays.append(photoperiod)
    data = xr.concat(arrays, dim=VARIABLE, coords="minimal")

    ordered_dims = tuple(dim for dim in ORDERED_CUBE_DIMS if dim in data.dims)
    data = data.transpose(*ordered_dims)

    time_limit = _option(product, "time_limit", defaults.preprocess.time_limit)
    time_selection = _option(product, "time_selection", defaults.preprocess.time_selection)
    data = _select_time(data, time_limit, selection=str(time_selection))

    min_valid_ratio = _option(product, "min_valid_ratio", defaults.preprocess.min_valid_ratio)
    if min_valid_ratio is not None and "cloud_mask" in data[VARIABLE].values and "land_mask" in data[VARIABLE].values:
        ratio = valid_water_coverage(data.sel({VARIABLE: "cloud_mask"}), data.sel({VARIABLE: "land_mask"}))
        data = data.isel(Id=(ratio >= float(min_valid_ratio)).values)

    fillna = _option(product, "fillna", defaults.preprocess.fillna)
    if fillna is not None:
        data = data.fillna(fillna)
    return data


def _target_floor_value(data: xr.DataArray, floor_config: dict[str, Any] | None) -> float | None:
    if not floor_config:
        return None

    method = str(floor_config.get("method", "")).lower()
    if method != "positive_quantile":
        raise ValueError("problem.target_floor.method must be 'positive_quantile'.")

    q = float(floor_config.get("q", floor_config.get("quantile", 0.01)))
    if q < 0.0 or q > 1.0:
        raise ValueError("problem.target_floor.q must be between 0 and 1.")

    values = np.asarray(data.values, dtype=float)
    positive_values = values[np.isfinite(values) & (values > 0)]
    if positive_values.size == 0:
        raise ValueError(
            "Cannot apply problem.target_floor because the target has no positive finite values."
        )

    floor = float(np.quantile(positive_values, q))
    if floor <= 0:
        raise ValueError("Computed problem.target_floor value must be greater than zero.")

    min_positive = floor_config.get("min_positive")
    if min_positive is not None:
        floor = max(floor, float(min_positive))
    return floor


def _apply_target_floor(
    data: xr.DataArray,
    floor_config: dict[str, Any] | None,
) -> tuple[xr.DataArray, float | dict[str, float] | None]:
    """Floor the target, one output at a time.

    Each output gets its own quantile: a shared floor would be dominated by whichever
    output has the largest magnitude, flooring the smaller ones far above their real range
    (mTAG percentages and cell counts differ by ~7 orders of magnitude, for instance).
    Returns a plain float for a single output so its recorded attrs stay as they were.
    """
    if not floor_config:
        return data, None
    if VARIABLE not in data.dims or data.sizes[VARIABLE] <= 1:
        floor = _target_floor_value(data, floor_config)
        if floor is None:
            return data, None
        return np.maximum(data, floor), floor

    names = [str(value) for value in data[VARIABLE].values]
    floors = {name: _target_floor_value(data.sel({VARIABLE: name}), floor_config) for name in names}
    if any(value is None for value in floors.values()):
        return data, None
    floor_array = xr.DataArray(
        [floors[name] for name in names], dims=(VARIABLE,), coords={VARIABLE: names}
    )
    return np.maximum(data, floor_array), floors


def _invalid_target_detail(data: xr.DataArray, invalid: xr.DataArray) -> str:
    """Name the offending output(s) so a vector target says which head to fix."""
    if VARIABLE not in data.dims or data.sizes[VARIABLE] <= 1:
        return ""
    counts = invalid.sum(dim=[dim for dim in invalid.dims if dim != VARIABLE])
    offenders = [
        f"{str(name)} ({int(count)})"
        for name, count in zip(data[VARIABLE].values, np.asarray(counts.values).reshape(-1))
        if int(count)
    ]
    return f" Offending output(s) and counts: {', '.join(offenders)}." if offenders else ""


def _target_floor_attrs(floor: float | dict[str, float], floor_config: dict[str, Any]) -> dict[str, Any]:
    """netCDF attrs can hold scalars, strings and numeric arrays -- not a mapping, so a
    per-output floor is recorded as two parallel entries instead of a dict."""
    attrs: dict[str, Any] = {
        "target_floor_method": "positive_quantile",
        "target_floor_q": float(floor_config.get("q", floor_config.get("quantile", 0.01))),
    }
    if isinstance(floor, dict):
        attrs["target_floor_variables"] = ",".join(floor)
        attrs["target_floor_value_after_offset"] = [float(value) for value in floor.values()]
    else:
        attrs["target_floor_value_after_offset"] = floor
    return attrs


@dataclass(frozen=True)
class TargetTransform:
    """How the caller wants the target column transformed before it is written to disk.

    Declared here, in the data layer, but populated by the caller: deciding that a target
    is modelled as log-of-relative-abundance is a modelling choice, so this module takes
    it as an argument instead of reaching into `config.problem`. That keeps this package
    usable without any modelling configuration at all.
    """

    transform: str = "none"
    offset: float = 0.0
    floor: dict[str, Any] | None = None

    @classmethod
    def none(cls) -> "TargetTransform":
        return cls()


def transform_target(
    data: xr.DataArray,
    transform: str,
    offset: float = 0.0,
    floor_config: dict[str, Any] | None = None,
) -> xr.DataArray:
    transform = transform.lower()
    if transform == "none":
        return data
    if transform == "log":
        shifted = data + offset
        shifted, floor = _apply_target_floor(shifted, floor_config)
        invalid = shifted <= 0
        invalid_count = int(invalid.sum().item())
        if invalid_count:
            min_value = float(data.min(skipna=True).item())
            raise ValueError(
                "Cannot apply problem.target_transform: log because the target contains "
                f"{invalid_count} values where target + offset is non-positive."
                f"{_invalid_target_detail(data, invalid)} "
                f"Minimum target value: {min_value}; offset: {offset}. Use a larger "
                "problem.target_transform_offset, set target_transform: none if the target "
                "is already logged, or filter/remove invalid target rows before preprocessing."
            )
        result = np.log(shifted)
        if floor is not None:
            result.attrs.update(_target_floor_attrs(floor, floor_config))
        return result
    if transform == "log1p":
        shifted = data + offset
        shifted, floor = _apply_target_floor(shifted, floor_config)
        invalid = shifted <= -1
        invalid_count = int(invalid.sum().item())
        if invalid_count:
            min_value = float(data.min(skipna=True).item())
            raise ValueError(
                "Cannot apply problem.target_transform: log1p because the target contains "
                f"{invalid_count} values where target + offset is less than or equal to -1."
                f"{_invalid_target_detail(data, invalid)} "
                f"Minimum target value: {min_value}; offset: {offset}."
            )
        result = np.log1p(shifted)
        if floor is not None:
            result.attrs.update(_target_floor_attrs(floor, floor_config))
        return result
    raise ValueError(f"Unsupported target transform: {transform}")


def preprocess_matchups(
    config: DataConfig,
    run_root: str | Path,
    target_transform: TargetTransform | None = None,
) -> dict[str, Path]:
    """Build the per-feature-group cubes and the target/metadata arrays for a run.

    `target_transform` is supplied by the caller (see `src/pipeline/preprocess.py`), which
    is the only layer entitled to know what the model intends to predict. Omitting it
    leaves the target untransformed.
    """
    run_root = Path(run_root)
    datasets_dir = run_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    target_transform = target_transform or TargetTransform.none()

    target_table_path = run_root / "processed" / "targets.csv"
    targets = pd.read_csv(target_table_path, parse_dates=["time"]) if target_table_path.exists() else load_target_table(config.target)

    target_da = target_to_dataarray(targets, target_names=config.target.target_names)
    target_da = transform_target(
        target_da,
        target_transform.transform,
        offset=target_transform.offset,
        floor_config=target_transform.floor,
    )
    target_path = datasets_dir / "target.nc"
    target_da.to_netcdf(target_path)

    meta = metadata_to_dataarray(
        targets,
        config.target.metadata_columns,
        include_spatial_metadata=config.target.include_spatial_metadata,
        include_day_metadata=config.target.include_day_metadata,
        include_cyclic_day_metadata=config.target.include_cyclic_day_metadata,
    )
    meta_path = datasets_dir / "meta.nc"
    meta.to_netcdf(meta_path)

    grouped: dict[str, list[xr.DataArray]] = defaultdict(list)
    artifacts: dict[str, Path] = {"target": target_path, "meta": meta_path}

    for product in config.products:
        matchup_path = run_root / "processed" / "matchups" / f"{product.name}.nc"
        if not matchup_path.exists():
            logger.warning(
                "Skipping product '%s': no matchup file at %s. Its feature group will be missing "
                "this product's channels. Check --run-id and that the matchup stage completed.",
                product.name,
                matchup_path,
            )
            continue
        ds = xr.load_dataset(matchup_path)
        group_name = product.feature_group or _option(product, "feature_group", None) or product.name
        grouped[group_name].append(_prepare_product_array(ds, product, config))

    groups: dict[str, xr.DataArray] = {}
    for group_name, arrays in grouped.items():
        arrays = _align_to_common_ids(arrays, group_name)
        if len(arrays) == 1:
            groups[group_name] = arrays[0]
        else:
            groups[group_name] = xr.concat(
                arrays, dim=VARIABLE, coords="minimal", compat="override", join="override"
            )

    if config.regrid.enabled:
        groups = _regrid_groups(groups, config.regrid)
    # Real per-sample coordinates (needed above for regridding) are only ever reset to
    # plain relative indices once, here, regardless of whether regridding ran.
    groups = {name: _use_relative_cube_coordinates(group) for name, group in groups.items()}

    for group_name, group in groups.items():
        path = datasets_dir / f"{group_name}.nc"
        group.to_netcdf(path)
        artifacts[group_name] = path
    return artifacts


def _regrid_groups(groups: dict[str, xr.DataArray], regrid: RegridConfig) -> dict[str, xr.DataArray]:
    reference_group = regrid.reference_group
    if reference_group not in groups:
        raise ValueError(
            f"regrid.reference_group '{reference_group}' is not one of the built feature "
            f"groups: {sorted(groups)}. Check config.regrid.reference_group and that its "
            "matchup/product entries produced a feature group of that name."
        )
    reference = groups[reference_group]
    reference_shape = (reference.sizes.get("lat"), reference.sizes.get("lon"))
    regridded = dict(groups)
    for group_name, group in groups.items():
        if group_name == reference_group:
            continue
        regridded[group_name] = resample_group_to_reference(group, reference, method=regrid.method)
    for group_name, group in regridded.items():
        shape = (group.sizes.get("lat"), group.sizes.get("lon"))
        if shape != reference_shape:
            raise ValueError(
                f"regrid: feature group '{group_name}' has lat/lon shape {shape} after "
                f"regridding, expected {reference_shape} (from reference group "
                f"'{reference_group}'). This indicates a bug in resample_group_to_reference, "
                "not a config problem."
            )
    return regridded
