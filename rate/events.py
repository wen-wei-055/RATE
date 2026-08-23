"""Access to the event archive.

One HDF5 file holds every event of the dataset, in the order

    [ train ... | val ... | test ... ]

with the two boundaries given in the config.  That ordering is what makes an
event's row index usable as an identity across training, retrieval and
evaluation, so it is checked rather than assumed wherever it can be.

Layout, as produced by the TEAM pre-processing (``japan.py``)::

    metadata/event_metadata     pandas table, one row per event
    metadata/sampling_rate      scalar
    metadata/pga_thresholds     the intensity levels warnings are issued for
    metadata/time_before        samples of noise in front of the first P pick
    data/<event name>/waveforms (stations, samples, channels)
    data/<event name>/coords    (stations, 4)  lat, lon, depth, borehole depth
    data/<event name>/p_picks   (stations,)    sample index of the P arrival
    data/<event name>/pga       (stations,)    log10 peak ground acceleration
    data/<event name>/pga_times (stations, thresholds)
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd


@dataclass
class Event:
    """One event's recordings, in the event's own station order."""

    waveforms: np.ndarray
    coords: np.ndarray
    p_picks: np.ndarray
    pga: np.ndarray


def detect_location_keys(columns) -> list[str]:
    """Find the latitude / longitude / depth columns of an event table.

    Datasets from different agencies name them differently; the first
    recognised spelling of each wins.
    """
    candidates = [
        ["LAT", "Latitude(\u00b0)", "Latitude", "source_latitude_deg"],
        ["LON", "Longitude(\u00b0)", "Longitude", "source_longitude_deg"],
        ["DEPTH", "JMA_Depth(km)", "Depth(km)", "Depth/Km", "source_depth_km"],
    ]
    keys = [next((key for key in group if key in columns), None) for group in candidates]
    if None in keys:
        raise ValueError(f"cannot find event coordinates among {list(columns)}")
    return keys


class EventStore:
    def __init__(self, path: str, event_key: str = "KiK_File", trace_length: int = 3000):
        self.path = path
        self.event_key = event_key
        self.trace_length = trace_length
        self.metadata = pd.read_hdf(path, "metadata/event_metadata")
        if event_key not in self.metadata.columns:
            raise KeyError(
                f"{path} has no event name column {event_key!r}; "
                f"available: {list(self.metadata.columns)}"
            )
        self.names = [str(name) for name in self.metadata[event_key]]
        with h5py.File(path, "r") as f:
            self.sampling_rate = int(f["metadata"]["sampling_rate"][()])
            self.pga_thresholds = f["metadata"]["pga_thresholds"][()]
            self.time_before = f["metadata"]["time_before"][()]
            missing = [n for n in self.names if n not in f["data"]]
        if missing:
            raise KeyError(f"{len(missing)} events listed in the metadata have no data, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.names)

    def origin_time(self, index: int) -> str:
        """The event's origin time, encoded in the first 14 chars of its name."""
        return self.names[index][:14]

    def read(self, handle: h5py.File, index: int) -> Event:
        group = handle["data"][self.names[index]]
        waveforms = group["waveforms"][:, : self.trace_length, :]
        if waveforms.shape[1] != self.trace_length:
            raise ValueError(
                f"event {self.names[index]} has {waveforms.shape[1]} samples, "
                f"expected at least trace_length={self.trace_length}"
            )
        return Event(
            waveforms=waveforms,
            coords=group["coords"][()],
            p_picks=group["p_picks"][()],
            pga=group["pga"][()],
        )

    def open(self) -> h5py.File:
        """Open the archive for reading; callers keep the handle for a while."""
        return h5py.File(self.path, "r")

    def station_counts(self, indices) -> np.ndarray:
        """How many stations recorded each event (read from shapes, not data)."""
        with h5py.File(self.path, "r") as f:
            return np.array([f["data"][self.names[i]]["pga"].shape[0] for i in indices])

    def pga_times(self, handle: h5py.File, index: int) -> np.ndarray:
        return handle["data"][self.names[index]]["pga_times"][()]

    def split(self, name: str, train_val_boundary: int, val_test_boundary: int) -> range:
        """Row indices belonging to one of ``train`` / ``val`` / ``test``."""
        bounds = {
            "train": (0, train_val_boundary),
            "val": (train_val_boundary, val_test_boundary),
            "test": (val_test_boundary, len(self)),
        }
        if name not in bounds:
            raise ValueError(f"unknown split {name!r}")
        start, stop = bounds[name]
        if start >= stop:
            raise ValueError(f"split {name!r} is empty (rows {start}:{stop} of {len(self)})")
        return range(start, stop)


class EventCache:
    """Recently read events, kept decoded.

    Evaluation re-reads the same event at every cutout time, and neighbouring
    events keep retrieving the same handful of historical ones, so a small
    cache turns a few hundred reads per event into a few.
    """

    def __init__(self, store: "EventStore", handle: h5py.File, size: int = 32):
        self.store = store
        self.handle = handle
        self.size = size
        self.events: dict[int, Event] = {}

    def read(self, index: int) -> Event:
        if index not in self.events:
            if len(self.events) >= self.size:
                self.events.pop(next(iter(self.events)))
            self.events[index] = self.store.read(self.handle, index)
        return self.events[index]
