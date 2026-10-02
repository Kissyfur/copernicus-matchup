from __future__ import annotations

from pathlib import Path

import xarray as xr

from copernicus_matchup.config import ProductSpec


def open_copernicus_dataset(product: ProductSpec) -> xr.Dataset:
    if product.source == "local":
        if not product.source_path:
            raise ValueError(f"Local product '{product.name}' requires source_path.")
        return xr.load_dataset(product.source_path)

    try:
        import copernicusmarine as cm
    except ImportError as exc:
        raise ImportError(
            "The download/matchup stages require the copernicusmarine package."
        ) from exc

    last_error: Exception | None = None
    for dataset_id in product.dataset_ids:
        try:
            return cm.open_dataset(dataset_id=dataset_id, **product.open_dataset_kwargs)
        except Exception as exc:  # pragma: no cover - depends on remote service
            last_error = exc
    raise RuntimeError(f"Could not open any dataset for product '{product.name}'.") from last_error


def rename_common_dimensions(ds: xr.Dataset, product: ProductSpec) -> xr.Dataset:
    dimension_rename_map = {
        old: new
        for old, new in product.rename_dimensions.items()
        if old in ds.dims or old in ds.coords or old in ds.data_vars
    }
    if dimension_rename_map:
        ds = ds.rename(dimension_rename_map)
    if product.variables:
        selected_variables = []
        missing = []
        for var in product.variables:
            renamed_var = product.rename_variables.get(var, var)
            if var in ds.data_vars:
                selected_variables.append(var)
            elif renamed_var in ds.data_vars:
                selected_variables.append(renamed_var)
            else:
                missing.append(var)
        if missing:
            raise ValueError(f"Product '{product.name}' is missing variables: {missing}")
        ds = ds[selected_variables]
    variable_rename_map = {
        old: new for old, new in product.rename_variables.items() if old in ds.data_vars
    }
    if variable_rename_map:
        ds = ds.rename(variable_rename_map)
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    return ds


def open_local_or_remote(product: ProductSpec, local_path: str | Path | None = None) -> xr.Dataset:
    if local_path is not None and Path(local_path).exists():
        return xr.load_dataset(local_path)
    if product.source == "local" and product.source_path:
        return rename_common_dimensions(xr.load_dataset(product.source_path), product)
    return rename_common_dimensions(open_copernicus_dataset(product), product)


def save_dataset(ds: xr.Dataset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path
