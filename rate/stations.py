"""The fixed station grid every event is projected onto.

An event only carries the stations that recorded it, in arbitrary order.  The
model on the other hand always sees the same ``n_stations`` slots, so each
recording has to be written into the slot its coordinates own.  ``StationGrid``
is the single place that mapping lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from obspy import UTCDateTime


class StationGrid:
    """Maps station coordinates to a fixed row index.

    The station JSON is ``{"lat,lon,depth,borehole_depth": row_index}``; the
    keys are matched by exact string, so the coordinates in the HDF5 file must
    be formatted the same way they were when the JSON was written.
    """

    def __init__(self, table: dict[str, int]):
        rows = sorted(table.values())
        if rows != list(range(len(table))):
            raise ValueError("station JSON must map to row indices 0..n-1 exactly once")
        self.table = table
        self.key_length = len(next(iter(table)).split(","))
        self.coords = np.array(
            [[float(v) for v in key.split(",")] for key in table], dtype="float64"
        )

    @classmethod
    def load(cls, path: str | Path) -> "StationGrid":
        with open(path) as f:
            return cls(json.load(f))

    def __len__(self) -> int:
        return len(self.table)

    def key(self, station_coords) -> str:
        return ",".join(f"{v}" for v in station_coords[: self.key_length])

    def rows_of(self, coords) -> np.ndarray:
        """Row index for every station of one event, in the event's own order."""
        return np.array([self.table[self.key(c)] for c in coords], dtype=int)


class StationAvailability:
    """Whether a station was in operation at a given time.

    Stations are installed and decommissioned over the life of a network, so a
    station that exists in the grid may not exist for a given event.  Those
    slots are excluded from the loss and zeroed in the predictions.
    """

    def __init__(self, grid: StationGrid, first: list[str] | None, last: list[str] | None):
        self.n_stations = len(grid)
        if (first is None) != (last is None):
            raise ValueError("station appearance windows must be given as a pair")
        if first is None:
            self.first = self.last = None
        else:
            if not len(first) == len(last) == self.n_stations:
                raise ValueError("appearance windows must cover every station in the grid")
            # Parsed once here rather than once per event: the original code
            # re-parsed n_events x n_stations timestamps on every data load.
            self.first = np.array([UTCDateTime(t).timestamp for t in first])
            self.last = np.array([UTCDateTime(t).timestamp for t in last])

    def usable(self, event_time: str) -> np.ndarray:
        """1.0 for stations in operation at ``event_time``, 0.0 otherwise."""
        if self.first is None:
            return np.ones(self.n_stations)
        t = UTCDateTime(event_time).timestamp
        return ((t >= self.first) & (t <= self.last)).astype("float64")


def load_time_table(path: str | Path | None) -> list[str] | None:
    """Read a ``{station_key: timestamp}`` JSON as a list in grid-row order."""
    if not path:
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with open(path) as f:
        return list(json.load(f).values())
