from __future__ import annotations

import numpy as np
import xarray as xr

from copernicus_matchup.targets import ID, LAT, LON


def resample_group_to_reference(
    source: xr.DataArray,
    reference: xr.DataArray,
    method: str = "linear",
) -> xr.DataArray:
    """Upsample `source` onto `reference`'s per-sample lat/lon grid.

    Both arrays must still carry the real per-sample absolute-degree lat/lon coordinates
    assigned by `_match_one` (dims (Id, lat) / (Id, lon)) -- this must run before
    preprocessing resets them to plain relative indices (`_use_relative_cube_coordinates`),
    otherwise there is no real grid left to interpolate onto. Samples present in only one
    array are dropped rather than silently misaligned.
    """
    reference_ids = set(reference[ID].values.tolist())
    common_ids = [id_ for id_ in source[ID].values if id_ in reference_ids]
    if not common_ids:
        raise ValueError(
            "resample_group_to_reference: no common Id values between the source and "
            "reference feature groups."
        )
    common_ids = np.asarray(common_ids, dtype=source[ID].values.dtype)

    source = source.sel(Id=common_ids)
    reference = reference.sel(Id=common_ids)

    resampled = []
    for id_value in common_ids:
        source_sample = source.sel(Id=[id_value])
        reference_sample = reference.sel(Id=[id_value])

        source_sample = source_sample.assign_coords(
            {
                LAT: (LAT, source_sample[LAT].isel(Id=0).values),
                LON: (LON, source_sample[LON].isel(Id=0).values),
            }
        )
        target_lat = reference_sample[LAT].isel(Id=0).values
        target_lon = reference_sample[LON].isel(Id=0).values

        # An axis with a single source point cannot be interpolated -- linear divides by a
        # zero coordinate span and returns all-NaN, and even "nearest" drops every target
        # point that falls outside that one point's bounds. It happens whenever a product's
        # grid is coarser than the matchup window (a 0.25-degree global model against a
        # 0.06-degree window gives exactly one cell), and it means the product reports one
        # value across the whole window. Broadcasting that value is what the data says;
        # axes with real structure are still interpolated normally.
        interp_axes = {}
        broadcast_axes = {}
        for dim, target in ((LAT, target_lat), (LON, target_lon)):
            if source_sample.sizes.get(dim, 0) >= 2:
                interp_axes[dim] = target
            else:
                broadcast_axes[dim] = target

        interpolated = source_sample
        if interp_axes:
            interpolated = interpolated.interp(method=method, **interp_axes)
        if broadcast_axes:
            interpolated = interpolated.reindex(method="nearest", **broadcast_axes)
        # Re-assign lat/lon as per-sample 2-D (Id, lat)/(Id, lon) coordinates, mirroring
        # `_match_one` (src/data/matchups.py) -- this keeps xr.concat(dim=Id) below from
        # treating lat/lon as a shared index it needs to align/merge across samples.
        interpolated = interpolated.assign_coords(
            {
                LAT: xr.DataArray([interpolated[LAT].values], dims=[ID, LAT]),
                LON: xr.DataArray([interpolated[LON].values], dims=[ID, LON]),
            }
        )
        resampled.append(interpolated)

    return xr.concat(resampled, dim=ID)
