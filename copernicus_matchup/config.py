from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from copernicus_matchup._utils import as_list


def _read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML configs require PyYAML. Install project requirements or use JSON."
            ) from exc
        data = yaml.safe_load(text)
        return {} if data is None else data
    raise ValueError(f"Unsupported config format: {path.suffix}")


DEFAULT_TARGET_NAME = "target"


def _parse_target_column(value: Any) -> str | list[str]:
    """Normalize target.target_column, preserving whether it was written as a list.

    A plain string stays a string (one output, unchanged behaviour); a list stays a list
    (a vector of outputs). Duplicates are rejected here rather than downstream, where they
    would silently collapse into one column when the target table is reshaped.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        columns = [str(item) for item in value]
        if not columns:
            raise ValueError("target.target_column must name at least one column.")
        duplicates = sorted({name for name in columns if columns.count(name) > 1})
        if duplicates:
            raise ValueError(f"target.target_column has duplicate entries: {duplicates}")
        return columns
    return str(value)


@dataclass
class TargetConfig:
    # A single name predicts one output; a list predicts a vector of outputs, one model
    # head per entry, in the order written here. See the target_names property for why a
    # single target keeps a different in-pipeline name than a vector does.
    path: str
    target_column: str | list[str]
    id_column: str = "Id"
    lat_column: str = "lat"
    lon_column: str = "lon"
    time_column: str = "time"
    sheet_name: str | int | None = None
    metadata_columns: list[str] = field(default_factory=list)
    include_spatial_metadata: bool = True
    include_day_metadata: bool = True
    include_cyclic_day_metadata: bool = True
    filter_column: str | None = None
    filter_values: list[str] = field(default_factory=list)

    @property
    def is_multi_output(self) -> bool:
        """True when target_column was written as a list, even a one-element one.

        Driven by how it was written rather than by how many entries it has, so
        `[ps_counts]` behaves like the vector case it looks like.
        """
        return not isinstance(self.target_column, str)

    @property
    def target_columns(self) -> list[str]:
        """Source column name(s) in the target table, always as a list."""
        return [self.target_column] if isinstance(self.target_column, str) else list(self.target_column)

    @property
    def target_names(self) -> list[str]:
        """The names these outputs carry through the pipeline.

        A single target keeps the historical "target" name so that target.nc and
        processed/targets.csv built by earlier runs stay readable and reusable -- runs are
        routinely rebuilt by copying another run's datasets/ directory. A vector keeps the
        source column names instead, so every artifact downstream (target.nc's variable
        axis, predictions.csv columns, per-output metrics) can say which output is which.
        """
        return self.target_columns if self.is_multi_output else [DEFAULT_TARGET_NAME]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetConfig":
        return cls(
            path=str(data["path"]),
            target_column=_parse_target_column(data["target_column"]),
            id_column=str(data.get("id_column", "Id")),
            lat_column=str(data.get("lat_column", "lat")),
            lon_column=str(data.get("lon_column", "lon")),
            time_column=str(data.get("time_column", "time")),
            sheet_name=data.get("sheet_name"),
            metadata_columns=[str(v) for v in data.get("metadata_columns", [])],
            include_spatial_metadata=bool(data.get("include_spatial_metadata", True)),
            include_day_metadata=bool(data.get("include_day_metadata", True)),
            include_cyclic_day_metadata=bool(data.get("include_cyclic_day_metadata", True)),
            filter_column=data.get("filter_column"),
            filter_values=[str(v) for v in data.get("filter_values", [])],
        )


@dataclass
class ProductSpec:
    name: str
    dataset_ids: list[str]
    source: str = "copernicus"
    source_path: str | None = None
    variables: list[str] = field(default_factory=list)
    feature_group: str | None = None
    open_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    rename_dimensions: dict[str, str] = field(
        default_factory=lambda: {"latitude": "lat", "longitude": "lon"}
    )
    rename_variables: dict[str, str] = field(default_factory=dict)
    matchup: dict[str, Any] = field(default_factory=dict)
    preprocess: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProductSpec":
        dataset_ids = data.get("dataset_ids", data.get("dataset_id"))
        return cls(
            name=str(data["name"]),
            dataset_ids=[str(v) for v in as_list(dataset_ids)],
            source=str(data.get("source", "copernicus")).lower(),
            source_path=str(data["source_path"]) if data.get("source_path") is not None else None,
            variables=[str(v) for v in data.get("variables", [])],
            feature_group=data.get("feature_group"),
            open_dataset_kwargs=dict(data.get("open_dataset_kwargs", {})),
            rename_dimensions=dict(
                data.get("rename_dimensions", {"latitude": "lat", "longitude": "lon"})
            ),
            rename_variables=dict(data.get("rename_variables", {})),
            matchup=dict(data.get("matchup", {})),
            preprocess=dict(data.get("preprocess", {})),
        )


@dataclass
class MatchupConfig:
    lat_window: float = 0.06
    lon_window: float = 0.06
    time_window_days: int = 1
    lat_threshold: float = 0.1
    lon_threshold: float = 0.1
    time_threshold_days: int = 1
    require_full_time_window: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MatchupConfig":
        data = {} if data is None else data
        return cls(
            lat_window=float(data.get("lat_window", 0.06)),
            lon_window=float(data.get("lon_window", 0.06)),
            time_window_days=int(data.get("time_window_days", 1)),
            lat_threshold=float(data.get("lat_threshold", 0.1)),
            lon_threshold=float(data.get("lon_threshold", 0.1)),
            time_threshold_days=int(data.get("time_threshold_days", 1)),
            require_full_time_window=bool(data.get("require_full_time_window", False)),
        )


@dataclass
class PreprocessConfig:
    positive_quantile: float | None = 0.01
    log_products: bool = True
    add_cloud_land_masks: bool = True
    fillna: float | None = 0.0
    min_valid_ratio: float | None = None
    time_limit: int | None = None
    time_selection: str = "first"
    prefix_variables: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PreprocessConfig":
        data = {} if data is None else data
        return cls(
            positive_quantile=data.get("positive_quantile", 0.01),
            log_products=bool(data.get("log_products", True)),
            add_cloud_land_masks=bool(data.get("add_cloud_land_masks", True)),
            fillna=data.get("fillna", 0.0),
            min_valid_ratio=data.get("min_valid_ratio"),
            time_limit=data.get("time_limit"),
            time_selection=str(data.get("time_selection", "first")).lower(),
            prefix_variables=bool(data.get("prefix_variables", False)),
        )


@dataclass
class RegridConfig:
    # Upsamples every other feature group onto `reference_group`'s per-sample lat/lon grid
    # before writing datasets/{group}.nc, so cnn3d's channel concatenation (which requires
    # identical lat/lon/time shapes across groups) can combine mismatched-resolution
    # products (e.g. optics at ~1km vs environment at ~4.2km). See docs/research.md.
    enabled: bool = False
    reference_group: str = ""
    method: str = "linear"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RegridConfig":
        data = {} if data is None else data
        allowed_keys = {"enabled", "reference_group", "method"}
        unexpected = sorted(set(data) - allowed_keys)
        if unexpected:
            raise ValueError(f"Unsupported regrid config key(s): {unexpected}")
        method = str(data.get("method", "linear")).lower()
        if method not in {"linear", "nearest"}:
            raise ValueError("regrid.method must be 'linear' or 'nearest'.")
        return cls(
            enabled=bool(data.get("enabled", False)),
            reference_group=str(data.get("reference_group", "")),
            method=method,
        )


@dataclass
class DataConfig:
    """Everything needed to turn in-situ observations into aligned feature cubes.

    This is the whole configuration surface of the acquisition half of the project: which
    observations to read, which Copernicus products to pull, how to match them in space
    and time, how to transform the channels, and what grid to resample onto. It says
    nothing about what will be predicted or how, which is what makes it separable from
    the modelling configuration -- see `ModelingConfig`.
    """

    target: TargetConfig
    products: list[ProductSpec]
    matchup: MatchupConfig = field(default_factory=MatchupConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    regrid: RegridConfig = field(default_factory=RegridConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        if "target" not in data:
            raise ValueError("Config must contain a 'target' section.")
        if "products" not in data or not data["products"]:
            raise ValueError("Config must contain at least one Copernicus product.")
        return cls(
            target=TargetConfig.from_dict(data["target"]),
            products=[ProductSpec.from_dict(p) for p in data["products"]],
            matchup=MatchupConfig.from_dict(data.get("matchup")),
            preprocess=PreprocessConfig.from_dict(data.get("preprocess")),
            regrid=RegridConfig.from_dict(data.get("regrid")),
        )


def load_data_config(path: str | Path) -> DataConfig:
    """Load a standalone acquisition config.

    Accepts either a flat mapping of sections or one wrapped in a `data:` block, so the
    same file can be read by this package alone or by a project that adds its own
    modelling sections alongside.
    """
    mapping = _read_mapping(Path(path))
    block = mapping.get("data")
    if isinstance(block, dict):
        mapping = {**{k: v for k, v in mapping.items() if k != "data"}, **block}
    return DataConfig.from_dict(mapping)
