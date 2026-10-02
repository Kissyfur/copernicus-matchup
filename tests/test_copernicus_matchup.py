"""Tests for copernicus_matchup.

Carried over from the project this package was extracted from, with imports repointed.
The bodies are unchanged: these are the regressions that justified the current behaviour,
including the clipped-window concat failure and the degenerate-axis broadcast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from copernicus_matchup.config import (
    DataConfig,
    MatchupConfig,
    PreprocessConfig,
    ProductSpec,
    RegridConfig,
    TargetConfig,
)
from copernicus_matchup.features.astronomy import photoperiod_hours
from copernicus_matchup.features.regrid import resample_group_to_reference
from copernicus_matchup.matchups import match_observations
from copernicus_matchup.preprocessing import (
    TargetTransform,
    _prepare_product_array,
    _select_time,
    preprocess_matchups,
    transform_target,
)
from copernicus_matchup.targets import (
    load_target_table,
    metadata_to_dataarray,
    target_to_dataarray,
)


def _write_target_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _per_sample_grid(ids, offsets) -> tuple[np.ndarray, np.ndarray]:
    # Mirrors _match_one's real per-sample absolute-degree window: each Id gets its own
    # base location (10 degrees apart) plus the same relative pixel offsets.
    coord = np.array([[10.0 * id_ + offset for offset in offsets] for id_ in ids])
    return coord, coord


def _per_sample_array(ids, offsets, var_names, value_fn, n_time=1) -> xr.DataArray:
    lat_coord, lon_coord = _per_sample_grid(ids, offsets)
    n, n_pix = len(ids), len(offsets)
    data = np.zeros((n, n_pix, n_pix, n_time, len(var_names)), dtype=np.float64)
    for i in range(n):
        for li, lat in enumerate(lat_coord[i]):
            for lo, lon in enumerate(lon_coord[i]):
                data[i, li, lo, :, :] = value_fn(lat, lon)
    return xr.DataArray(
        data,
        dims=("Id", "lat", "lon", "time", "variable"),
        coords={
            "Id": list(ids),
            "lat": (("Id", "lat"), lat_coord),
            "lon": (("Id", "lon"), lon_coord),
            "time": np.arange(n_time),
            "variable": list(var_names),
        },
    )


def test_target_config_parses_and_validates_vector_targets():
    single = TargetConfig.from_dict({"path": "t.csv", "target_column": "ps_counts"})
    assert not single.is_multi_output
    assert single.target_columns == ["ps_counts"]
    # A single target keeps the historical in-pipeline name, so target.nc and
    # processed/targets.csv built by earlier runs stay readable.
    assert single.target_names == ["target"]

    vector = TargetConfig.from_dict({"path": "t.csv", "target_column": ["ps_counts", "perc"]})
    assert vector.is_multi_output
    assert vector.target_columns == vector.target_names == ["ps_counts", "perc"]

    # An explicit one-element list is still the vector case: it names its output.
    one = TargetConfig.from_dict({"path": "t.csv", "target_column": ["ps_counts"]})
    assert one.is_multi_output and one.target_names == ["ps_counts"]

    with pytest.raises(ValueError, match="duplicate entries"):
        TargetConfig.from_dict({"path": "t.csv", "target_column": ["a", "a"]})
    with pytest.raises(ValueError, match="at least one column"):
        TargetConfig.from_dict({"path": "t.csv", "target_column": []})


def test_regrid_config_parses_and_validates_method():
    default = RegridConfig.from_dict(None)
    assert default.enabled is False
    assert default.reference_group == ""
    assert default.method == "linear"

    custom = RegridConfig.from_dict({"enabled": True, "reference_group": "optics", "method": "nearest"})
    assert custom.enabled is True
    assert custom.reference_group == "optics"
    assert custom.method == "nearest"

    with pytest.raises(ValueError, match="linear.*nearest"):
        RegridConfig.from_dict({"method": "cubic"})


def test_target_table_to_dataarray_keeps_configured_output_order(tmp_path):
    from copernicus_matchup.targets import load_target_table, target_to_dataarray

    path = tmp_path / "targets.csv"
    pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "lat": [40.0, 41.0, 42.0],
            "lon": [2.0, 3.0, 4.0],
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "counts": [10.0, 20.0, 30.0],
            "perc": [0.1, 0.2, 0.3],
        }
    ).to_csv(path, index=False)

    # Deliberately the reverse of the CSV's own column order: the variable axis must
    # follow the config, because every downstream artifact is positional.
    config = TargetConfig.from_dict({"path": str(path), "target_column": ["perc", "counts"]})
    array = target_to_dataarray(load_target_table(config), target_names=config.target_names)

    assert list(array["variable"].values) == ["perc", "counts"]
    assert array.shape == (3, 2)
    np.testing.assert_allclose(array.sel(variable="counts").values, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(array.sel(variable="perc").values, [0.1, 0.2, 0.3])

    missing = TargetConfig.from_dict({"path": str(path), "target_column": ["counts", "nope"]})
    with pytest.raises(ValueError, match="missing configured target column"):
        load_target_table(missing)


def test_metadata_preserves_missing_values_and_adds_cyclic_day():
    data = pd.DataFrame(
        {
            "Id": [1, 2],
            "time": pd.to_datetime(["2024-01-01", "2024-07-01"]),
            "tem": [15.0, np.nan],
            "sal": [37.0, 38.0],
        }
    )

    meta = metadata_to_dataarray(
        data,
        metadata_columns=["tem", "sal"],
        include_spatial_metadata=False,
        include_day_metadata=False,
        include_cyclic_day_metadata=True,
    )

    assert list(meta.coords["variable"].values) == ["x_day", "y_day", "tem", "sal"]
    assert np.isnan(meta.sel(Id=2, variable="tem").item())


def test_metadata_can_be_empty_for_environment_only_runs():
    data = pd.DataFrame(
        {
            "Id": [1, 2],
            "time": pd.to_datetime(["2024-01-01", "2024-07-01"]),
            "lat": [41.0, 42.0],
            "lon": [2.0, 3.0],
        }
    )

    meta = metadata_to_dataarray(
        data,
        metadata_columns=[],
        include_spatial_metadata=False,
        include_day_metadata=False,
        include_cyclic_day_metadata=False,
    )

    assert meta.dims == ("Id", "variable")
    assert meta.shape == (2, 0)
    assert meta["Id"].values.tolist() == [1, 2]
    assert meta["variable"].values.tolist() == []


def test_log_target_transform_supports_offset():
    target = xr.DataArray([0.0, 900.0], dims="Id", coords={"Id": [1, 2]})
    transformed = transform_target(target, "log", offset=100.0)

    assert np.allclose(transformed.values, np.log([100.0, 1000.0]))


def test_log_target_transform_supports_positive_quantile_floor():
    target = xr.DataArray([0.0, 1.0, 9.0, np.nan], dims="Id", coords={"Id": [1, 2, 3, 4]})
    transformed = transform_target(
        target,
        "log",
        floor_config={"method": "positive_quantile", "q": 0.0},
    )

    assert np.allclose(transformed.values[:3], np.log([1.0, 1.0, 9.0]))
    assert np.isnan(transformed.values[3])
    assert transformed.attrs["target_floor_method"] == "positive_quantile"
    assert transformed.attrs["target_floor_q"] == 0.0
    assert transformed.attrs["target_floor_value_after_offset"] == 1.0


def test_target_floor_is_computed_per_output():
    """A shared floor would be set by whichever output has the largest magnitude."""
    from copernicus_matchup.preprocessing import transform_target

    # counts ~1e4, perc ~1e-3: a pooled quantile would floor perc far above its own range.
    data = xr.DataArray(
        np.array([[1e4, 1e-3], [2e4, 2e-3], [3e4, 3e-3], [0.0, 0.0]]),
        dims=("Id", "variable"),
        coords={"Id": [1, 2, 3, 4], "variable": ["counts", "perc"]},
    )
    result = transform_target(data, "log", floor_config={"method": "positive_quantile", "q": 0.01})

    floors = dict(
        zip(
            result.attrs["target_floor_variables"].split(","),
            result.attrs["target_floor_value_after_offset"],
        )
    )
    assert floors["counts"] == pytest.approx(1e4, rel=0.05)
    assert floors["perc"] == pytest.approx(1e-3, rel=0.05)
    # The zero row is floored to each output's own floor, not to a shared one.
    np.testing.assert_allclose(
        np.exp(result.sel(Id=4).values), [floors["counts"], floors["perc"]], rtol=0.05
    )


def test_time_selection_can_use_days_before_matchup_center():
    data = xr.DataArray(
        np.arange(29),
        dims=("time",),
        coords={"time": pd.date_range("2024-01-01", periods=29)},
    )

    selected = _select_time(data, limit=8, selection="past_to_center")

    assert selected.values.tolist() == list(range(7, 15))
    assert selected.time.values[-1] == data.time.values[14]


def test_time_selection_can_use_symmetric_days_around_matchup_center():
    data = xr.DataArray(
        np.arange(29),
        dims=("time",),
        coords={"time": pd.date_range("2024-01-01", periods=29)},
    )

    selected = _select_time(data, limit=29, selection="centered")

    assert selected.values.tolist() == list(range(29))
    assert selected.time.values[14] == data.time.values[14]


def test_photoperiod_hours_matches_known_reference_values():
    # Equator: ~12h year-round.
    assert photoperiod_hours(0.0, 1) == pytest.approx(12.0, abs=0.01)
    assert photoperiod_hours(0.0, 266) == pytest.approx(12.0, abs=0.01)
    # High-latitude solstices clamp to the polar day/night bounds.
    assert photoperiod_hours(70.0, 172) == pytest.approx(24.0)
    assert photoperiod_hours(70.0, 355) == pytest.approx(0.0)
    # Southern hemisphere is antisymmetric: their winter solstice is the north's summer.
    assert photoperiod_hours(-70.0, 172) == pytest.approx(0.0)
    # Vectorized over arrays.
    result = photoperiod_hours(np.array([0.0, 70.0]), np.array([172, 172]))
    assert result[0] == pytest.approx(12.0, abs=0.5)
    assert result[1] == pytest.approx(24.0)


def test_resample_group_to_reference_matches_shape_and_interpolates():
    ids = [1, 2]
    # source ("coarse"): 2x2 pixels per sample, spanning [base, base+1] in both lat/lon.
    source = _per_sample_array(ids, [0.0, 1.0], ["v"], lambda lat, lon: 100 * lat + lon)
    # reference ("fine"): 3x3 pixels per sample, strictly within the source's span, so
    # linear interpolation never needs to extrapolate.
    reference = _per_sample_array(ids, [0.2, 0.5, 0.8], ["v"], lambda lat, lon: 0.0)

    result = resample_group_to_reference(source, reference, method="linear")

    assert result.sizes["lat"] == 3
    assert result.sizes["lon"] == 3
    assert result.sizes["Id"] == 2
    # Bilinear interpolation of an exactly linear/bilinear field (100*lat + lon) recovers
    # the analytically exact value at every target point -- a correctness check, not just
    # a shape check.
    ref_lat, ref_lon = _per_sample_grid(ids, [0.2, 0.5, 0.8])
    for i in range(2):
        for li in range(3):
            for lo in range(3):
                expected = 100 * ref_lat[i, li] + ref_lon[i, lo]
                actual = float(result.isel(Id=i, lat=li, lon=lo, time=0).sel(variable="v").values)
                assert actual == pytest.approx(expected)


def test_resample_group_to_reference_drops_ids_missing_from_either_group():
    source = _per_sample_array([1, 2, 3], [0.0, 1.0], ["v"], lambda lat, lon: lat + lon)
    reference = _per_sample_array([2, 3, 4], [0.2, 0.8], ["v"], lambda lat, lon: 0.0)

    result = resample_group_to_reference(source, reference, method="linear")

    assert sorted(result["Id"].values.tolist()) == [2, 3]


def test_regrid_broadcasts_a_product_coarser_than_the_matchup_window():
    """A 0.25-degree global product against a 0.06-degree window yields ONE cell per
    sample. Interpolating from a single point is undefined -- linear divided by a zero
    coordinate span and returned all-NaN, silently emptying every environment channel.
    """
    ids = [1]
    # one source cell, as a coarse global model gives; three reference cells, as 4km optics gives
    source = xr.DataArray(
        np.full((1, 1, 1, 2, 1), 5.0),
        dims=("Id", "lat", "lon", "time", "variable"),
        coords={
            "Id": ids,
            "lat": xr.DataArray([[40.0]], dims=["Id", "lat"]),
            "lon": xr.DataArray([[10.0]], dims=["Id", "lon"]),
            "time": [0, 1],
            "variable": ["thetao"],
        },
    )
    reference = xr.DataArray(
        np.zeros((1, 3, 3, 2, 1)),
        dims=("Id", "lat", "lon", "time", "variable"),
        coords={
            "Id": ids,
            "lat": xr.DataArray([[39.96, 40.0, 40.04]], dims=["Id", "lat"]),
            "lon": xr.DataArray([[9.96, 10.0, 10.04]], dims=["Id", "lon"]),
            "time": [0, 1],
            "variable": ["412"],
        },
    )

    resampled = resample_group_to_reference(source, reference, method="linear")

    assert resampled.sizes["lat"] == 3 and resampled.sizes["lon"] == 3
    assert not np.isnan(resampled.values).any(), "a coarse product must broadcast, not vanish"
    np.testing.assert_allclose(resampled.values, 5.0)


def test_regrid_still_interpolates_an_axis_that_has_real_structure():
    """Only the degenerate axis is broadcast; a resolved axis is interpolated as before."""
    ids = [1]
    source = xr.DataArray(
        np.array([5.0, 7.0]).reshape(1, 2, 1, 1, 1),
        dims=("Id", "lat", "lon", "time", "variable"),
        coords={
            "Id": ids,
            "lat": xr.DataArray([[39.9, 40.1]], dims=["Id", "lat"]),
            "lon": xr.DataArray([[10.0]], dims=["Id", "lon"]),  # single point -> broadcast
            "time": [0],
            "variable": ["thetao"],
        },
    )
    reference = xr.DataArray(
        np.zeros((1, 3, 3, 1, 1)),
        dims=("Id", "lat", "lon", "time", "variable"),
        coords={
            "Id": ids,
            "lat": xr.DataArray([[39.9, 40.0, 40.1]], dims=["Id", "lat"]),
            "lon": xr.DataArray([[9.96, 10.0, 10.04]], dims=["Id", "lon"]),
            "time": [0],
            "variable": ["412"],
        },
    )

    resampled = resample_group_to_reference(source, reference, method="linear")

    assert not np.isnan(resampled.values).any()
    # lat was interpolated (5 -> 6 -> 7), lon was broadcast (identical across lon)
    lat_profile = resampled.isel(Id=0, lon=0, time=0, variable=0).values
    np.testing.assert_allclose(lat_profile, [5.0, 6.0, 7.0])
    for lon_index in range(3):
        np.testing.assert_allclose(
            resampled.isel(Id=0, lon=lon_index, time=0, variable=0).values, lat_profile
        )


def test_matchups_drop_windows_clipped_by_the_product_domain():
    """An observation outside the product's domain snaps to an edge cell and gets a
    smaller window, which xarray cannot concatenate with the rest -- it used to raise
    "cannot reindex or align along dimension 'lat'". Those observations belong in the
    unmatched list: a clipped window means the product does not cover that sample.
    """
    from copernicus_matchup.config import MatchupConfig
    from copernicus_matchup.matchups import match_observations

    lats = np.round(np.arange(40.0, 40.5, 0.0417), 4)
    lons = np.round(np.arange(10.0, 10.5, 0.0417), 4)
    times = pd.date_range("2020-01-01", periods=3)
    region = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.ones((len(times), len(lats), len(lons))))},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    observations = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            # two comfortably inside the domain, one past its northern edge
            "lat": [40.2, 40.25, float(lats[-1]) + 0.02],
            "lon": [10.2, 10.2, 10.2],
            "time": [times[1], times[1], times[1]],
        }
    )
    config = MatchupConfig(
        lat_window=0.06, lon_window=0.06, time_window_days=1,
        lat_threshold=10, lon_threshold=10, time_threshold_days=1,
    )

    matched, unmatched = match_observations(observations, region, config)

    # The edge observation is reported as unmatched rather than crashing the concat.
    assert list(unmatched["Id"]) == [3]
    assert matched is not None
    assert list(matched["Id"].values) == [1, 2]
    # And the surviving window is the full, uniform one.
    assert matched.sizes["lat"] == 3


def test_load_target_table_filters_to_configured_values(tmp_path):
    path = tmp_path / "targets.csv"
    _write_target_csv(
        path,
        [
            {"ID": "a", "lat": 1.0, "lon": 1.0, "time": "2020-01-01", "y": 1.0, "SF": "0.2-3 µm"},
            {"ID": "b", "lat": 2.0, "lon": 2.0, "time": "2020-01-02", "y": 2.0, "SF": "3-20 µm"},
            {"ID": "c", "lat": 3.0, "lon": 3.0, "time": "2020-01-03", "y": 3.0, "SF": "0.2-3 µm"},
        ],
    )
    config = TargetConfig(
        path=str(path),
        target_column="y",
        id_column="ID",
        filter_column="SF",
        filter_values=["0.2-3 µm"],
    )

    data = load_target_table(config)

    assert sorted(data["Id"]) == ["a", "c"]


def test_load_target_table_missing_filter_column_raises(tmp_path):
    path = tmp_path / "targets.csv"
    _write_target_csv(
        path,
        [{"ID": "a", "lat": 1.0, "lon": 1.0, "time": "2020-01-01", "y": 1.0}],
    )
    config = TargetConfig(
        path=str(path),
        target_column="y",
        id_column="ID",
        filter_column="SF",
        filter_values=["0.2-3 µm"],
    )

    with pytest.raises(ValueError, match="SF"):
        load_target_table(config)


def test_load_target_table_without_filter_keeps_all_rows(tmp_path):
    path = tmp_path / "targets.csv"
    _write_target_csv(
        path,
        [
            {"ID": "a", "lat": 1.0, "lon": 1.0, "time": "2020-01-01", "y": 1.0, "SF": "0.2-3 µm"},
            {"ID": "b", "lat": 2.0, "lon": 2.0, "time": "2020-01-02", "y": 2.0, "SF": "3-20 µm"},
        ],
    )
    config = TargetConfig(path=str(path), target_column="y", id_column="ID")

    data = load_target_table(config)

    assert sorted(data["Id"]) == ["a", "b"]


def test_preprocess_matchups_builds_cubes_from_a_data_config_alone(tmp_path):
    """The whole point of the split: cubes are built with no modelling config in sight.

    Exercises the full chain -- regrid onto the reference group, per-product transforms,
    the target transform with its quantile floor -- driven only by a `DataConfig`.
    """
    run_root = tmp_path / "run"
    matchups_dir = run_root / "processed" / "matchups"
    matchups_dir.mkdir(parents=True)

    ids = list(range(1, 9))
    _per_sample_array(ids, [0.0, 1.0], ["nh4", "no3"], lambda la, lo: la + lo, n_time=3).to_dataset(
        dim="variable"
    ).to_netcdf(matchups_dir / "nutrients.nc")
    _per_sample_array(ids, [0.2, 0.5, 0.8], ["refl"], lambda la, lo: 2.0 + 0.01 * la, n_time=3).to_dataset(
        dim="variable"
    ).to_netcdf(matchups_dir / "reflectance.nc")

    pd.DataFrame(
        {
            "Id": ids,
            "lat": [10.0 * i for i in ids],
            "lon": [10.0 * i for i in ids],
            "time": pd.date_range("2020-01-01", periods=len(ids)),
            "target": [0.001 * i for i in ids],
        }
    ).to_csv(run_root / "processed" / "targets.csv", index=False)

    config = DataConfig(
        target=TargetConfig(path=str(run_root / "unused.csv"), target_column="target"),
        products=[
            ProductSpec(
                name="nutrients",
                dataset_ids=["d"],
                variables=["nh4", "no3"],
                feature_group="nut",
                preprocess={"log1p": True, "interpolate_dims": ["lat", "lon", "time"]},
            ),
            ProductSpec(
                name="reflectance",
                dataset_ids=["d"],
                variables=["refl"],
                feature_group="optics",
                preprocess={
                    "log": True,
                    "interpolate_dims": ["lat", "lon", "time"],
                    "add_cloud_land_masks": True,
                },
            ),
        ],
        preprocess=PreprocessConfig(
            time_limit=2, time_selection="centered", fillna=0.0,
            # masks are derived from the ocean-colour product only, which is how the
            # real configs use them: off globally, enabled on the reflectance product
            add_cloud_land_masks=False,
        ),
        regrid=RegridConfig(enabled=True, reference_group="optics", method="linear"),
    )

    artifacts = preprocess_matchups(
        config,
        run_root,
        target_transform=TargetTransform(
            transform="log", floor={"method": "positive_quantile", "q": 0.01}
        ),
    )

    assert set(artifacts) == {"target", "meta", "nut", "optics"}
    nut = xr.load_dataarray(artifacts["nut"])
    optics = xr.load_dataarray(artifacts["optics"])
    # the coarse group is resampled onto the reference group's grid
    assert (nut.sizes["lat"], nut.sizes["lon"]) == (optics.sizes["lat"], optics.sizes["lon"]) == (3, 3)
    assert nut.sizes["time"] == optics.sizes["time"] == 2
    assert [str(v) for v in nut["variable"].values] == ["nh4", "no3"]
    # the mask channels are appended after the data channels, on the product that asked
    assert [str(v) for v in optics["variable"].values] == ["refl", "cloud_mask", "land_mask"]

    target = xr.load_dataarray(artifacts["target"])
    assert target.attrs["target_floor_method"] == "positive_quantile"
    assert np.isfinite(target.values).all()


def test_data_config_round_trips_through_yaml(tmp_path):
    """A standalone acquisition config loads from disk with no modelling sections."""
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "target:\n"
        "  path: obs.csv\n"
        "  target_column: y\n"
        "products:\n"
        "  - name: reflectance\n"
        "    dataset_ids: [some-dataset]\n"
        "    variables: [RRS412]\n"
        "    feature_group: optics\n"
        "matchup:\n"
        "  lat_window: 0.06\n"
        "preprocess:\n"
        "  time_limit: 15\n"
        "  time_selection: centered\n",
        encoding="utf-8",
    )

    from copernicus_matchup import load_data_config

    config = load_data_config(config_path)

    assert config.target.target_column == "y"
    assert [p.name for p in config.products] == ["reflectance"]
    assert config.matchup.lat_window == 0.06
    assert config.preprocess.time_limit == 15
    assert config.preprocess.time_selection == "centered"


def test_data_config_also_accepts_a_wrapping_data_block(tmp_path):
    """The grouped form, as written by a project that adds its own sections alongside."""
    config_path = tmp_path / "grouped.yaml"
    config_path.write_text(
        "data:\n"
        "  target:\n"
        "    path: obs.csv\n"
        "    target_column: y\n"
        "  products:\n"
        "    - name: reflectance\n"
        "      dataset_ids: [some-dataset]\n"
        "      variables: [RRS412]\n"
        "  preprocess:\n"
        "    time_limit: 9\n"
        "modeling:\n"
        "  model:\n"
        "    family: random_forest\n",
        encoding="utf-8",
    )

    from copernicus_matchup import load_data_config

    config = load_data_config(config_path)

    assert config.preprocess.time_limit == 9
    assert [p.name for p in config.products] == ["reflectance"]
