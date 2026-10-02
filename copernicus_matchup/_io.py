from __future__ import annotations

from pathlib import Path

import xarray as xr

#: Every artifact this package writes is NETCDF4, stated explicitly rather than left to
#: xarray's engine search.
#:
#: Without it, the format depends on which backend happens to be installed: with netCDF4
#: present xarray writes NETCDF4 (HDF5), but if it falls back to scipy it writes NetCDF3,
#: and the two are not interchangeable. NetCDF3 permits only one unlimited dimension and
#: only at index 0, so an array with a zero-length second dimension -- which is exactly
#: what the metadata array is for a run that requests no metadata columns -- gets written
#: with the unlimited dimension in an illegal position and cannot be read back at all
#: ("NetCDF: NC_UNLIMITED in the wrong index"). The same data under NETCDF4 round-trips
#: fine. Pinning the format makes the artifacts reproducible across environments instead
#: of silently dependent on the install.
NETCDF_FORMAT = "NETCDF4"


def write_netcdf(data: xr.DataArray | xr.Dataset, path: str | Path) -> Path:
    """Write `data` to `path` in this package's fixed NetCDF format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(path, format=NETCDF_FORMAT)
    return path
