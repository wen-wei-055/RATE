"""Batch construction.

Both datasets yield whole batches (the DataLoader is used only to iterate and
shuffle batch order, with ``batch_size=None``), shaped

    waveforms  (batch, stations * (1 + retrieved), samples, channels)
    coords     (batch, stations * (1 + retrieved), 4)
    pga        (batch, stations)          target, current event only
    usable     (batch, stations)          1 where the station was in operation

With ``retrieved_events = 0`` the second block is absent and the batch is
exactly what stage 1 (pre-training) used, which is the whole reason the two
training stages can share one implementation.
"""

from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .events import Event, EventStore
from .retrieval import RetrievalIndex
from .stations import StationAvailability, StationGrid

#: Where a retrieved event's first P arrival is placed, in samples.  Retrieved
#: waveforms are aligned to this so that "time since the event started" means
#: the same thing in the current and in the retrieved block.
RETRIEVED_PICK_OFFSET = 500


def identity_collate(batch):
    """The dataset already returns batches; keep them as they are."""
    return batch


class _GridBatch:
    """Scratch space for one batch, in station-grid coordinates."""

    def __init__(self, batch_size: int, grid: StationGrid, config: DataConfig, retrieved: int):
        n = len(grid)
        blocks = 1 + retrieved
        self.grid = grid
        self.config = config
        self.n_stations = n
        self.waveforms = np.zeros((batch_size, n * blocks, config.trace_length, config.channels))
        self.coords = np.tile(grid.coords, (batch_size, blocks, 1))
        self.pga = np.zeros((batch_size, n))
        self.usable = np.zeros((batch_size, n))

    def fill_block(
        self,
        block: int,
        events: list[Event | None],
        cutout: int,
        visible_until: int,
        blind: bool,
    ) -> None:
        """Write one block of stations for the whole batch.

        ``cutout`` is where the current event is cut; ``visible_until`` is where
        *this* block is cut, which for a retrieved event reaches further so the
        model can see how the historical event went on to develop.
        """
        config = self.config
        n = self.n_stations
        waveforms = np.zeros((len(events), n, config.trace_length, config.channels))
        p_picks = np.zeros((len(events), n))

        for i, event in enumerate(events):
            if event is None:
                continue
            rows = self.grid.rows_of(event.coords)
            waveforms[i, rows] = event.waveforms
            p_picks[i, rows] = event.p_picks
            if block == 0:
                self.pga[i, rows] = np.where(np.isinf(event.pga), 0.0, event.pga)

        if config.adjust_mean:
            waveforms -= np.mean(waveforms[:, :, : cutout + 1], axis=2, keepdims=True)
        waveforms[:, :, visible_until:] = 0

        if config.trigger_based:
            # A station that has not triggered yet must not leak its waveform.
            p_picks[p_picks <= 0] = config.trace_length
            waveforms[cutout < p_picks, :, :] = 0

        if config.integrate:
            waveforms = np.cumsum(waveforms, axis=2) / config.sampling_rate

        if blind:
            blind_random_stations(waveforms)

        self.waveforms[:, block * n : (block + 1) * n] = waveforms

    def as_tensors(self):
        inputs = (
            torch.from_numpy(self.waveforms.astype("float32")),
            torch.from_numpy(self.coords.astype("float32")),
        )
        targets = (
            torch.from_numpy(self.pga.astype("float32")),
            torch.from_numpy(self.usable.astype("float32")),
        )
        return inputs, targets


def blind_random_stations(waveforms: np.ndarray) -> None:
    """Zero a random subset of the recording stations, in place.

    Augmentation: a real network is missing arbitrary stations at any moment,
    and the model has to stay useful when it is.
    """
    mask = np.zeros(waveforms.shape[:2], dtype=bool)
    for i in range(waveforms.shape[0]):
        active = np.where((waveforms[i] != 0).any(axis=(1, 2)))[0]
        if len(active) == 0:
            active = np.zeros(1, dtype=int)
        blind_length = np.random.randint(0, len(active))
        np.random.shuffle(active)
        mask[i, active[:blind_length]] = True
    waveforms[mask] = 0


def epoch_indices(store: EventStore, rows: range, config: DataConfig) -> np.ndarray:
    """Positions within ``rows`` to train on, oversampling the rare events.

    Large earthquakes and events recorded by many stations are heavily
    outnumbered, so they are repeated instead of reweighted.
    """
    magnitudes = np.asarray(store.metadata[config.magnitude_key].values[rows.start : rows.stop])
    indices = np.arange(len(rows))
    if config.min_magnitude is not None:
        indices = indices[magnitudes >= config.min_magnitude]

    if config.magnitude_resampling > 1:
        base = indices
        for m in np.arange(config.min_upsample_magnitude, 9):
            in_bin = base[(m < magnitudes[base]) & (magnitudes[base] <= m + 1)]
            repeats = int(config.magnitude_resampling ** (m - 1) - 1)
            indices = np.concatenate((indices, np.repeat(in_bin, repeats)))

    if config.upsample_high_station_events:
        counts = store.station_counts([rows.start + i for i in indices])
        repeats = counts // config.upsample_high_station_events + 1
        indices = np.repeat(indices, repeats)

    return np.repeat(indices, config.oversample)


class TrainDataset(Dataset):
    """Batches of training or validation events, read lazily from the archive."""

    def __init__(
        self,
        store: EventStore,
        split: str,
        config: DataConfig,
        availability: StationAvailability,
        grid: StationGrid,
        retrieval: RetrievalIndex | None = None,
        limit: int | None = None,
    ):
        self.store = store
        self.config = config
        self.grid = grid
        self.retrieval = retrieval
        self.retrieved = 1 if retrieval is not None else 0

        rows = store.split(split, config.train_val_boundary, config.val_test_boundary)
        if limit:
            rows = range(rows.start, min(rows.stop, rows.start + limit))
        self.rows = rows
        self.usable = np.stack([availability.usable(store.origin_time(r)) for r in rows])

        self.indexes = epoch_indices(store, rows, config)
        np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        return int(np.ceil(len(self.indexes) / self.config.batch_size))

    def __getitem__(self, index: int):
        config = self.config
        cutout = int(np.random.randint(*config.cutout))
        offsets = self.indexes[index * config.batch_size : (index + 1) * config.batch_size]

        with h5py.File(self.store.path, "r") as f:
            current = [self.store.read(f, self.rows.start + o) for o in offsets]
            retrieved = []
            if self.retrieval is not None:
                step = self.retrieval.time_step(cutout, config.sampling_rate)
                for o in offsets:
                    picked = self.retrieval.pick(self.rows.start + o, step)
                    retrieved.append(None if picked < 0 else align_retrieved(self.store.read(f, picked)))

        batch = _GridBatch(len(offsets), self.grid, config, self.retrieved)
        batch.usable[:] = self.usable[offsets]
        batch.fill_block(0, current, cutout, cutout, blind=config.station_blinding)
        if self.retrieval is not None:
            batch.fill_block(
                1, retrieved, cutout, cutout + config.peek_samples, blind=config.station_blinding
            )
        return batch.as_tensors()


class EvalDataset(Dataset):
    """One event, cut at a series of increasing times.

    Item ``i`` is the batch of size one the model would see ``times[i]`` seconds
    after the start of the record, which is what warning times are measured
    against.  No augmentation is applied here.
    """

    def __init__(
        self,
        store: EventStore,
        handle,
        event_index: int,
        times,
        config: DataConfig,
        grid: StationGrid,
        usable: np.ndarray,
        retrieval: RetrievalIndex | None = None,
        neighbours_recorded: int = 10,
    ):
        self.store = store
        self.handle = handle
        self.event_index = event_index
        self.times = times
        self.config = config
        self.grid = grid
        self.usable = usable
        self.retrieval = retrieval
        self.retrieved = 1 if retrieval is not None else 0
        self.neighbours_recorded = neighbours_recorded

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, index: int):
        config = self.config
        cutout = int(config.sampling_rate * (self.times[index] + config.noise_seconds))

        neighbours = np.full(self.neighbours_recorded, -1)
        retrieved = None
        if self.retrieval is not None:
            step = self.retrieval.time_step(cutout, config.sampling_rate)
            neighbours = self.retrieval.neighbours(self.event_index, step, self.neighbours_recorded)
            if neighbours[0] >= 0:
                retrieved = align_retrieved(self.store.read(self.handle, neighbours[0]))

        batch = _GridBatch(1, self.grid, config, self.retrieved)
        batch.usable[0] = self.usable
        current = self.store.read(self.handle, self.event_index)
        batch.fill_block(0, [current], cutout, cutout, blind=False)
        if self.retrieval is not None:
            batch.fill_block(1, [retrieved], cutout, cutout + config.peek_samples, blind=False)
        inputs, (pga, usable) = batch.as_tensors()
        return inputs, (pga, usable, neighbours)


def align_retrieved(event: Event) -> Event:
    """Shift a historical event so its first P arrival sits at a known sample."""
    return replace(event, p_picks=event.p_picks - (np.min(event.p_picks) - RETRIEVED_PICK_OFFSET))
