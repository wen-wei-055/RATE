"""Builds the retrieval database.

    python -m rate.preprocess --config configs/rate.json

Each event is reduced to one vector per elapsed second: the peak ground
acceleration seen so far at every station of the grid, normalised to unit
length so that an inner product is a cosine similarity.  Stations that did not
record the event stay at zero, which is what makes the vector a *spatial*
signature rather than a magnitude estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .config import Config
from .events import EventStore
from .stations import StationGrid


def peak_acceleration(waveform: np.ndarray) -> np.ndarray:
    """Running vector magnitude of the three surface components."""
    return np.sqrt(np.sum(waveform[:, :3] ** 2, axis=-1))


def build(store: EventStore, grid: StationGrid, time_steps: int, sampling_rate: int) -> np.ndarray:
    """(events, stations, time steps + 1) of unit-length signatures.

    Time step 0 is left empty: it stands for "nothing observed yet", which no
    query ever asks about.
    """
    signatures = np.zeros((len(store), len(grid), time_steps + 1))
    with h5py.File(store.path, "r") as f:
        for index in tqdm(range(len(store)), desc="signatures"):
            event = store.read(f, index)
            rows = grid.rows_of(event.coords)
            for station, row in enumerate(rows):
                pga = peak_acceleration(event.waveforms[station])
                for step in range(1, time_steps + 1):
                    signatures[index, row, step] = np.max(pga[: step * sampling_rate])

    norm = np.sqrt(np.sum(signatures**2, axis=1, keepdims=True))
    signatures = np.divide(signatures, norm, out=np.zeros_like(signatures), where=norm > 0)
    return signatures.astype("float32")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, help="defaults to the config's retrieval database path")
    parser.add_argument("--time-steps", type=int, default=30, help="seconds of history to index")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    output = Path(args.output or config.data.retrieval.database)
    if not output.name:
        raise ValueError("no output path: set --output or data.retrieval.database")

    grid = StationGrid.load(config.data.station_json)
    store = EventStore(config.data.data_path, config.data.event_key, config.model.trace_length)
    signatures = build(store, grid, args.time_steps, config.data.sampling_rate)
    np.save(output, signatures)
    print(f"wrote {output} {signatures.shape}")


if __name__ == "__main__":
    main()
