from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Where a run's acquisition artifacts live, relative to its root directory.

    Only the directories this package writes are named here. A consuming project is free
    to put its own outputs (models, metrics, reports) under the same root; the names
    below are the contract between the two.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def raw(self) -> Path:
        """Whole downloaded products, for local sources, plus markers for remote ones."""
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        """The standardized target table."""
        return self.root / "processed"

    @property
    def targets(self) -> Path:
        return self.processed / "targets.csv"

    @property
    def matchups(self) -> Path:
        """One NetCDF of target-centred windows per product, plus its unmatched rows."""
        return self.processed / "matchups"

    @property
    def datasets(self) -> Path:
        """The aligned cubes, one per feature group, plus `target` and `meta`."""
        return self.root / "datasets"

    def product_matchup(self, name: str) -> Path:
        return self.matchups / f"{name}.nc"

    def product_unmatched(self, name: str) -> Path:
        return self.matchups / f"{name}_unmatched.csv"

    def raw_product(self, name: str) -> Path:
        return self.raw / f"{name}.nc"

    def remote_marker(self, name: str) -> Path:
        return self.raw / f"{name}.remote.json"

    def mkdirs(self, *names: str) -> None:
        for name in names:
            getattr(self, name).mkdir(parents=True, exist_ok=True)
