"""A tiny synthetic dataset, so the pipeline can be exercised without the real one.

Nothing here is seismologically meaningful: the point is that the shapes, the
station grid, the retrieval database and the HDF5 layout are the real ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

N_STATIONS = 8
N_EVENTS = 12
TRACE_LENGTH = 600
SAMPLING_RATE = 100
CHANNELS = 6
THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2]
TIME_BEFORE = 2


def station_table() -> dict[str, int]:
    return {
        f"{35.0 + 0.1 * i},{137.0 + 0.1 * i},{-1.0 - i},{0.005}": i for i in range(N_STATIONS)
    }


def write(directory: Path, seed: int = 0) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stations = station_table()
    coords = np.array([[float(v) for v in key.split(",")] for key in stations])

    names = [f"2012010100{i:02d}00" for i in range(N_EVENTS)]
    metadata = pd.DataFrame(
        {
            "KiK_File": names,
            "M_J": rng.uniform(3.0, 7.0, N_EVENTS),
            "LAT": rng.uniform(34.5, 36.0, N_EVENTS),
            "LON": rng.uniform(136.5, 138.0, N_EVENTS),
            "DEPTH": rng.uniform(5.0, 50.0, N_EVENTS),
        }
    )

    data_path = directory / "synthetic.hdf5"
    metadata.to_hdf(data_path, key="metadata/event_metadata", mode="w", format="table")
    with h5py.File(data_path, "a") as f:
        meta = f["metadata"]
        meta.create_dataset("sampling_rate", data=SAMPLING_RATE)
        meta.create_dataset("pga_thresholds", data=np.array(THRESHOLDS))
        meta.create_dataset("time_before", data=TIME_BEFORE)
        group = f.create_group("data")
        for i, name in enumerate(names):
            recording = sorted(rng.choice(N_STATIONS, size=N_STATIONS - 2, replace=False))
            n = len(recording)
            event = group.create_group(name)
            waveforms = rng.normal(0, 1e-3, (n, TRACE_LENGTH, CHANNELS))
            picks = rng.integers(150, 300, n)
            for station in range(n):
                waveforms[station, picks[station]:] *= 50
            event.create_dataset("waveforms", data=waveforms)
            event.create_dataset("coords", data=coords[recording])
            event.create_dataset("p_picks", data=picks)
            event.create_dataset("pga", data=rng.uniform(-2.0, 0.5, n))
            # Not read by this code; the archives carry it, so the fixture does.
            event.create_dataset("pgv", data=rng.uniform(-3.0, 0.0, n))
            event.create_dataset(
                "pga_times", data=rng.integers(250, 500, (n, len(THRESHOLDS)))
            )

    station_json = directory / "stations.json"
    station_json.write_text(json.dumps(stations, indent=4))
    first = directory / "first_appearance.json"
    last = directory / "last_appearance.json"
    first.write_text(json.dumps({k: "2010-01-01T00:00:00.000000Z" for k in stations}, indent=4))
    last.write_text(json.dumps({k: "2020-01-01T00:00:00.000000Z" for k in stations}, indent=4))
    return {"data": data_path, "stations": station_json, "first": first, "last": last}


def configs(directory: Path, paths: dict[str, Path]) -> dict[str, Path]:
    """Matching pre-training and RATE configs for the synthetic dataset."""
    model = {
        "stations": N_STATIONS,
        "retrieved_events": 0,
        "trace_length": TRACE_LENGTH,
        "borehole": True,
        "downsample": 1,
        "mlp_dims": [20, 20, 20],
        "output_mlp_dims": [10, 5],
        "wavelength": [[0.01, 15], [0.01, 15], [0.01, 10]],
        "rotation": None,
        "rotation_anchor": [35, 0],
        "transformer_layers": 1,
        "n_heads": 2,
        "ffn_hidden_dim": 20,
        "pga_mixture": 2,
        "legacy_shared_qkv": True,
    }
    data = {
        "data_path": str(paths["data"]),
        "station_json": str(paths["stations"]),
        "first_station_appearance": str(paths["first"]),
        "last_station_appearance": str(paths["last"]),
        "train_val_boundary": 6,
        "val_test_boundary": 9,
        "batch_size": 2,
        "sampling_rate": SAMPLING_RATE,
        "noise_seconds": 2,
        "cutout_start": 0,
        "cutout_end": 3,
        "magnitude_resampling": 1.0,
    }
    pretrain = {
        "seed": 0,
        "model": model,
        "data": data,
        "training": {"weight_path": str(directory / "pretrain"), "epochs": 2, "device": "cpu",
                     "torch_threads": 1, "keep_all_until": 1},
    }
    rate = {
        "seed": 0,
        "model": {**model, "retrieved_events": 1},
        "data": {
            **data,
            "adjust_mean": False,
            "retrieval": {
                "database": str(directory / "signatures.npy"),
                "topk": 1,
                "peek_seconds": 1.0,
                "exclude_self": True,
            },
        },
        "training": {
            "weight_path": str(directory / "rate"),
            "epochs": 2,
            "device": "cpu",
            "torch_threads": 1,
            "keep_all_until": 1,
            "transfer_model_path": str(directory / "pretrain"),
            "freeze": ["TotalEmbedding", "To_GaussianDistribution"],
        },
    }
    written = {}
    for name, config in (("pretrain", pretrain), ("rate", rate)):
        path = directory / f"{name}.json"
        path.write_text(json.dumps(config, indent=4))
        written[name] = path
    return written
