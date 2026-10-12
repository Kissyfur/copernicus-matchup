"""Download a daily Copernicus Marine dataset over regional boxes and average it by month.

    python scripts/download_monthly_means.py configs/med_radiometry_monthly.yaml [--dry-run]

For each region in the config, writes <output_dir>/<region>_monthly.nc. It holds one mean per
calendar month for every variable, over the days with a valid value, stamped with the first day
of the month. `valid_days` is the number of days averaged in each pixel. The optional
`positive_quantile` and `log` settings then transform the means as the matchup config's options
of the same name do. Needs the Copernicus Marine client and a login (`copernicusmarine login`).

The download runs in blocks aligned with the server's time chunks, about five years each, read
from the catalogue. Aligned blocks transfer about 1.1x the data, against up to 6x for
calendar-year blocks. Each finished block is kept as monthly sums and counts under
<output_dir>/parts, so an interrupted run resumes at the first unfinished block, and later runs
only fetch the newest block.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import copernicusmarine as cm
import numpy as np
import xarray as xr
import yaml

logger = logging.getLogger("monthly_means")


def _as_date(value) -> dt.date | None:
    if value is None or isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.datetime):
        return value.date()
    return dt.date.fromisoformat(str(value))


def _ms_to_date(ms: float) -> dt.date:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date()


def dataset_time_layout(dataset_id: str) -> tuple[dt.date, dt.date, int | None]:
    """First day, last day and time-chunk length (in days) of the dataset's time-series service."""
    catalogue = cm.describe(dataset_id=dataset_id, disable_progress_bar=True)
    part = catalogue.products[0].datasets[0].versions[0].parts[0]
    first = last = chunk_days = None
    for service in part.services:
        if "series" not in str(service.service_name):
            continue
        time = next(c for c in service.variables[0].coordinates if c.coordinate_id == "time")
        first, last = _ms_to_date(time.minimum_value), _ms_to_date(time.maximum_value)
        if "time-series" in str(service.service_name):
            chunk_days = int(time.chunking_length * time.step / 86_400_000)
    if first is None:
        raise RuntimeError(f"Dataset {dataset_id} has no ARCO service to read its time axis from.")
    return first, last, chunk_days


def blocks(first: dt.date, chunk_days: int | None, start: dt.date, end: dt.date):
    """(block_start, block_end, complete) covering [start, end], aligned with the server's chunks.

    `complete` is False for a block cut short by `end`; it is the only one a later run, with
    more data available, has to fetch again.
    """
    if not chunk_days:
        yield start, end, False
        return
    k = (start - first).days // chunk_days
    while True:
        block_start = first + dt.timedelta(days=k * chunk_days)
        block_end = block_start + dt.timedelta(days=chunk_days - 1)
        if block_start > end:
            return
        yield max(block_start, start), min(block_end, end), block_end <= end
        k += 1


def monthly_sums(daily_path: Path, variables: list[str]) -> xr.Dataset:
    """Per-month sum and count of valid values, so blocks that split a month can be combined."""
    with xr.open_dataset(daily_path, chunks="auto") as daily:
        daily = daily[variables]
        sums = daily.resample(time="MS").sum(skipna=True, keep_attrs=True)
        counts = daily.notnull().resample(time="MS").sum().astype("int16")
        part = xr.merge(
            [
                sums.rename({v: f"{v}_sum" for v in variables}),
                counts.rename({v: f"{v}_count" for v in variables}),
            ]
        )
        return part.compute()


def combine(parts: xr.Dataset, variables: list[str], attrs: dict) -> xr.Dataset:
    """Monthly means from the per-block sums and counts, concatenated along time."""
    # A month cut by a block boundary appears in two parts; summing per time merges it.
    totals = parts.groupby("time").sum(keep_attrs=True)
    monthly = xr.Dataset(attrs=attrs)
    counts = []
    for v in variables:
        count = totals[f"{v}_count"]
        monthly[v] = (totals[f"{v}_sum"] / count.where(count > 0)).astype("float32")
        # Only what holds for every date: the source's other attributes (sensor lists, valid
        # ranges) describe one sample file.
        source_attrs = totals[f"{v}_sum"].attrs
        monthly[v].attrs = {k: source_attrs[k] for k in ("long_name", "standard_name", "units") if k in source_attrs}
        counts.append(count)
    # Bands share the cloud mask almost exactly (RRS412 misses a few more days), so one count
    # per pixel is reported: the days behind the best-covered variable.
    monthly["valid_days"] = xr.concat(counts, dim="variable").max("variable").astype("int16")
    monthly["valid_days"].attrs = {"long_name": "number of days averaged", "units": "1"}
    monthly["time"].attrs = {"long_name": "first day of the month averaged"}
    return monthly


def transform(monthly: xr.Dataset, variables: list[str], quantile: float | None, log: bool) -> xr.Dataset:
    """The matchup config's `positive_quantile` and `log`, applied to the monthly means.

    Per variable, over all months and pixels: values below the `quantile` of the positive
    means are raised to it, then the natural log is taken -- in that order, so the log never
    sees a value below the floor. The floor is recorded in the variable's attributes.
    """
    for v in variables:
        band = monthly[v].load()
        attrs = dict(band.attrs)
        if quantile is not None:
            floor = float(band.where(band > 0).quantile(quantile, skipna=True))
            band = np.maximum(band, floor)   # NaN (no valid day) stays NaN
            attrs.update(positive_quantile_q=quantile, positive_quantile_floor=floor)
        if log:
            band = np.log(band.where(band > 0))
            attrs["long_name"] = f"natural log of {attrs.get('long_name', v)}"
            if "units" in attrs:
                attrs["units"] = f"log({attrs['units']})"
        monthly[v] = band.astype("float32")
        monthly[v].attrs = attrs
    return monthly


def _write(data: xr.Dataset, path: Path, level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": level} for name in data.data_vars}
    tmp = path.with_suffix(".tmp.nc")
    data.to_netcdf(tmp, format="NETCDF4", encoding=encoding)
    tmp.replace(path)


def _part_is_current(path: Path, request: str) -> bool:
    if not path.exists():
        return False
    with xr.open_dataset(path) as part:
        return part.attrs.get("request") == request


def process_region(config: dict, name: str, box: dict, layout: tuple, dry_run: bool) -> Path | None:
    first, last, chunk_days = layout
    start = max(_as_date(config.get("start")) or first, first)
    end = min(_as_date(config.get("end")) or last, last)
    variables = list(config["variables"])
    lat, lon = box["lat"], box["lon"]
    output_dir = Path(config.get("output_dir", "outputs/monthly_means"))
    parts_dir, daily_dir = output_dir / "parts" / name, output_dir / "daily"
    # A part is reused only if it was made for exactly this request.
    request = json.dumps(
        {"dataset_id": config["dataset_id"], "variables": variables, "lat": lat, "lon": lon},
        sort_keys=True,
    )
    subset_args = dict(
        dataset_id=config["dataset_id"],
        variables=variables,
        minimum_latitude=lat[0],
        maximum_latitude=lat[1],
        minimum_longitude=lon[0],
        maximum_longitude=lon[1],
        disable_progress_bar=True,
    )

    part_paths, transfer_mb = [], 0.0
    for block_start, block_end, complete in blocks(first, chunk_days, start, end):
        label = f"{name} {block_start}..{block_end}"
        part_path = parts_dir / f"{block_start}_{block_end}.nc"
        part_paths.append(part_path)
        if _part_is_current(part_path, request):
            logger.info("%s: already done", label)
            continue
        window = dict(start_datetime=f"{block_start}T00:00:00", end_datetime=f"{block_end}T00:00:00")
        if dry_run:
            response = cm.subset(**subset_args, **window, dry_run=True)
            block_mb = float(str(response.data_transfer_size).split()[0])
            transfer_mb += block_mb
            logger.info("%s: would transfer %.0f MB", label, block_mb)
            continue

        # The newest block grows between runs: drop its outdated versions.
        for stale in [*parts_dir.glob(f"{block_start}_*.nc"), *daily_dir.glob(f"{name}_{block_start}_*.nc")]:
            stale.unlink()
        logger.info("%s: downloading", label)
        response = cm.subset(
            **subset_args,
            **window,
            output_directory=daily_dir,
            output_filename=f"{name}_{block_start}_{block_end}.nc",
            netcdf_compression_level=1,
            overwrite=True,
        )
        daily_path = Path(response.file_path)
        logger.info("%s: averaging by month", label)
        part = monthly_sums(daily_path, variables)
        part.attrs["request"] = request
        _write(part, part_path, level=1)
        if not config.get("keep_daily", False):
            daily_path.unlink()
        if not complete:
            logger.info("%s: partial block, fetched again if the end date moves", label)

    if dry_run:
        logger.info("%s: about %.1f GB to transfer", name, transfer_mb / 1024)
        return None

    quantile = config.get("positive_quantile")
    log = bool(config.get("log", False))
    method = "mean of the valid daily values of each calendar month; time is its first day"
    if quantile is not None:
        method += f"; per variable, values below the {quantile} quantile of the positive means raised to it"
    if log:
        method += "; natural log"
    attrs = {
        "title": f"Monthly mean of {config['dataset_id']}",
        "source_dataset": config["dataset_id"],
        "region": name,
        "latitude_range": f"{lat[0]} to {lat[1]}",
        "longitude_range": f"{lon[0]} to {lon[1]}",
        "period": f"{start} to {end}",
        "method": method,
    }
    output = output_dir / f"{name}_monthly.nc"
    # Opened lazily, so the parts stream through instead of being held in memory at once.
    with xr.open_mfdataset(part_paths, combine="nested", concat_dim="time", data_vars="all") as parts:
        monthly = transform(combine(parts, variables, attrs), variables, quantile, log)
        _write(monthly, output, level=4)
    logger.info("%s: wrote %s", name, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="YAML config with dataset_id, variables and regions.")
    parser.add_argument("--region", action="append", help="Only this region; repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Report the transfer size per block, download nothing.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    # The Copernicus client logs every request; its warnings are enough here.
    logging.getLogger("copernicusmarine").setLevel(logging.WARNING)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    regions = config["regions"]
    unknown = sorted(set(args.region or []) - set(regions))
    if unknown:
        parser.error(f"unknown region(s): {unknown}; the config has {sorted(regions)}")

    layout = dataset_time_layout(config["dataset_id"])
    logger.info(
        "%s: %s to %s, server time chunks of %s days",
        config["dataset_id"], layout[0], layout[1], layout[2],
    )
    for name in args.region or regions:
        process_region(config, name, regions[name], layout, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
