from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from copernicus_matchup._io import write_netcdf
from copernicus_matchup.config import MatchupConfig, ProductSpec
from copernicus_matchup.copernicus import open_local_or_remote, rename_common_dimensions
from copernicus_matchup.targets import ID, LAT, LON, TIME

logger = logging.getLogger("copernicus_matchup.matchups")


def _timedelta_days(days: int) -> pd.Timedelta:
    return pd.Timedelta(days=days)


def _nearest_time_distance(value, target) -> pd.Timedelta:
    return abs(pd.Timestamp(value) - pd.Timestamp(target))


def _match_one(
    obs_id,
    lat: float,
    lon: float,
    date,
    region: xr.Dataset,
    config: MatchupConfig,
) -> xr.Dataset | None:
    near_point = region.sel({TIME: date, LAT: lat, LON: lon}, method="nearest")
    near_date = near_point[TIME].values
    near_lat = float(near_point[LAT].values)
    near_lon = float(near_point[LON].values)

    if (
        _nearest_time_distance(near_date, date) >= _timedelta_days(config.time_threshold_days)
        or abs(near_lon - lon) >= config.lon_threshold
        or abs(near_lat - lat) >= config.lat_threshold
    ):
        return None

    match = region.sel(
        {
            TIME: slice(
                pd.Timestamp(near_date) - _timedelta_days(config.time_window_days),
                pd.Timestamp(near_date) + _timedelta_days(config.time_window_days),
            ),
            LAT: slice(near_lat - config.lat_window, near_lat + config.lat_window),
            LON: slice(near_lon - config.lon_window, near_lon + config.lon_window),
        }
    )
    if config.require_full_time_window:
        expected = 2 * config.time_window_days + 1
        if TIME in match.sizes and match.sizes[TIME] != expected:
            return None

    if "depth" in match.coords or "depth" in match.dims:
        match = match.sel(depth=match.depth.min())

    lats_coord = xr.DataArray([match[LAT].values], dims=[ID, LAT])
    lons_coord = xr.DataArray([match[LON].values], dims=[ID, LON])
    time_coord = xr.DataArray([match[TIME].values], dims=[ID, TIME])
    return match.assign_coords({LON: lons_coord, LAT: lats_coord, TIME: time_coord, ID: (ID, [obs_id])})


def _drop_clipped_windows(matches: list[xr.Dataset], unmatched: list) -> list[xr.Dataset]:
    """Keep only the windows with the most common lat/lon size.

    An observation whose window runs off the edge of the product's domain gets a clipped,
    smaller patch -- 2 grid cells instead of 3, say. Those cannot be concatenated with the
    rest (xarray raises "conflicting dimension sizes"), and they are not salvageable
    either: a clipped window means the sample sits outside the area this product covers,
    so the values it did match are the domain's edge cells rather than the sample's own
    surroundings. They are recorded as unmatched, exactly like a sample no product covers.
    """
    if not matches:
        return matches
    shapes = [(match.sizes.get(LAT), match.sizes.get(LON)) for match in matches]
    # Most common wins, ties broken toward the larger window: clipping can only remove
    # cells, so with an even split the bigger shape is the intact one. Count comes first
    # so an irregular grid, where a few windows legitimately run large, cannot outvote the
    # shape most observations actually got.
    modal_shape = max(
        set(shapes), key=lambda shape: (shapes.count(shape), shape[0] * shape[1])
    )
    kept = []
    for match, shape in zip(matches, shapes):
        if shape == modal_shape:
            kept.append(match)
        else:
            unmatched.extend(match[ID].values.tolist())
    if len(kept) != len(matches):
        logger.warning(
            "Dropped %d of %d matchups whose lat/lon window was clipped to %s instead of "
            "%s -- those observations fall outside this product's domain.",
            len(matches) - len(kept),
            len(matches),
            sorted({shape for shape in shapes if shape != modal_shape}),
            modal_shape,
        )
    return kept


def match_observations(
    observations: pd.DataFrame,
    region: xr.Dataset,
    config: MatchupConfig,
) -> tuple[xr.Dataset | None, pd.DataFrame]:
    matches = []
    unmatched = []
    region = region.sortby(LAT).sortby(LON)
    for row in observations.itertuples(index=False):
        matched = _match_one(
            getattr(row, ID),
            float(getattr(row, LAT)),
            float(getattr(row, LON)),
            getattr(row, TIME),
            region,
            config,
        )
        if matched is None:
            unmatched.append(getattr(row, ID))
        else:
            matches.append(matched)
    matches = _drop_clipped_windows(matches, unmatched)
    if not matches:
        return None, observations[observations[ID].isin(unmatched)].copy()
    return xr.concat(matches, dim=ID), observations[observations[ID].isin(unmatched)].copy()


def product_matchup_config(product: ProductSpec, default: MatchupConfig) -> MatchupConfig:
    if not product.matchup:
        return default
    data = {name: getattr(default, name) for name in MatchupConfig.__dataclass_fields__}
    data.update(product.matchup)
    return MatchupConfig.from_dict(data)


def create_product_matchups(
    product: ProductSpec,
    observations: pd.DataFrame,
    config: MatchupConfig,
    raw_path: str | Path | None = None,
) -> tuple[xr.Dataset | None, pd.DataFrame]:
    region = open_local_or_remote(product, raw_path)
    region = rename_common_dimensions(region, product)
    return match_observations(observations, region, product_matchup_config(product, config))


def save_matchups(matchups: xr.Dataset | None, unmatched: pd.DataFrame, matchup_path: Path, unmatched_path: Path) -> None:
    unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    unmatched.to_csv(unmatched_path, index=False)
    if matchups is not None:
        matchup_path.parent.mkdir(parents=True, exist_ok=True)
        matchups.load()
        write_netcdf(matchups, matchup_path)
