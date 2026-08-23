"""Warning-time evaluation.

    python -m rate.evaluate --experiment-path weight_path --checkpoint 45

For every event of the split, the model is re-run at a series of increasing
cutout times.  A station is "warned" as soon as the predicted probability of
exceeding an intensity threshold passes alpha; comparing that moment with the
moment the threshold was actually exceeded gives the warning time, which is
what an early warning system is judged on.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from geopy.distance import geodesic
from scipy.stats import norm
from tqdm import tqdm

from .config import Config
from .data import EvalDataset
from .events import EventCache, EventStore, detect_location_keys
from .model import FullModel
from .retrieval import RetrievalIndex
from .stations import StationAvailability, StationGrid, load_time_table

G = 9.81


def load_model(config: Config, weight_file: Path, device) -> FullModel:
    model = FullModel(config.model)
    model.load_state_dict(torch.load(weight_file, map_location="cpu")["model_weights"])
    return model.to(device).eval()


@torch.no_grad()
def predict_cutouts(model: FullModel, dataset: EvalDataset, device):
    """Run one event at every cutout time.

    Returns the mixture parameters, ``(times, stations, components, 3)``, and
    the retrieved neighbours per cutout.  Stations that were out of service
    come back as NaN, which propagates into "never warned".
    """
    predictions, neighbours = [], []
    for i in range(len(dataset)):
        (waveforms, coords), (_, usable, retrieved) = dataset[i]
        usable = usable.to(device)
        out = model.predict_current(waveforms.to(device), coords.to(device), usable)[0]
        out[usable[0] == 0] = 0
        predictions.append(out.cpu().numpy())
        neighbours.append(retrieved)

    predictions = np.stack(predictions)
    with np.errstate(invalid="ignore"):
        predictions[..., 0] /= np.sum(predictions[..., 0], axis=-1, keepdims=True)
    return predictions, np.stack(neighbours)


def first_warning_times(predictions: np.ndarray, times: np.ndarray, thresholds, alpha) -> np.ndarray:
    """When each station is first warned, per threshold and per alpha.

    ``(stations, thresholds, alphas)`` in seconds, NaN where no warning is ever
    issued.
    """
    first = np.zeros((predictions.shape[1], len(thresholds), len(alpha)), dtype=int)
    for j, level in enumerate(np.log10(np.asarray(thresholds) * G)):
        probability = np.sum(
            predictions[:, :, :, 0]
            * (1 - norm.cdf((level - predictions[:, :, :, 1]) / predictions[:, :, :, 2])),
            axis=-1,
        )
        exceedance = probability[:, :, None] > np.asarray(alpha)
        # A leading "not warned" row makes argmax return 0 when it never is.
        exceedance = np.pad(exceedance, ((1, 0), (0, 0), (0, 0)), mode="constant")
        first[:, j] = np.argmax(exceedance, axis=0)

    first -= 1
    seconds = np.full(first.shape, np.nan)
    seconds[first > -1] = times[first[first > -1]]
    return seconds


def true_exceedance_times(store: EventStore, handle, cache, grid: StationGrid, index: int,
                          n_thresholds: int):
    """When each station actually exceeded each threshold, in seconds."""
    times = np.zeros((len(grid), n_thresholds), dtype=float)
    times[grid.rows_of(cache.read(index).coords)] = store.pga_times(handle, index)
    times[times == 0] = np.nan
    return times / store.sampling_rate - store.time_before


def hypocentral_distances(grid: StationGrid, epicentre) -> np.ndarray:
    """Distance from the hypocentre to every station of the grid, in km."""
    surface = np.array([geodesic(station[:2], epicentre[:2]).km for station in grid.coords])
    return np.sqrt(surface**2 + epicentre[2] ** 2)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-path", type=str, required=True, help="a training run's weight_path")
    parser.add_argument("--checkpoint", type=str, required=True, help="epoch number, e.g. 45")
    parser.add_argument("--val", action="store_true", help="evaluate the validation split instead of the test split")
    parser.add_argument("--blind-time", type=float, default=0.5, help="first evaluation time after the P arrival")
    parser.add_argument("--last-time", type=float, default=25.0)
    parser.add_argument("--time-step", type=float, default=0.2)
    parser.add_argument("--alpha", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--torch-threads", type=int, default=5)
    parser.add_argument(
        "--ignore-station-windows",
        action="store_true",
        help="treat every station as in service, as the old evaluate.py did unless it was "
             "given --first_station_appearance_path / --last_station_appearance_path",
    )
    args = parser.parse_args(argv)

    experiment_path = Path(args.experiment_path)
    config = Config.load(experiment_path / "config.json")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")

    split = "val" if args.val else "test"
    suffix = "" if args.blind_time == 0.5 else f"_blind{args.blind_time:.1f}"
    output_dir = experiment_path / f"evaluation{suffix}" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = StationGrid.load(config.data.station_json)
    windows = (None, None)
    if not args.ignore_station_windows:
        windows = (
            load_time_table(config.data.first_station_appearance),
            load_time_table(config.data.last_station_appearance),
        )
    availability = StationAvailability(grid, *windows)
    store = EventStore(config.data.data_path, config.data.event_key, config.model.trace_length)
    if store.time_before != config.data.noise_seconds:
        print(
            f"warning: the dataset keeps {store.time_before}s of noise before the P arrival "
            f"but the config says noise_seconds={config.data.noise_seconds}"
        )
    retrieval = RetrievalIndex.build(config.data.retrieval, pool=config.data.train_val_boundary)
    model = load_model(config, experiment_path / f"checkpoint_{args.checkpoint}.pth", device)

    coord_keys = detect_location_keys(store.metadata.columns)
    times = np.arange(args.blind_time, args.last_time, args.time_step)
    alpha = [float(x) for x in args.alpha.split(",")]
    events = store.split(split, config.data.train_val_boundary, config.data.val_test_boundary)

    results, retrieved_per_event = [], []
    with store.open() as handle:
        cache = EventCache(store, handle)
        for index in tqdm(events, desc=f"{split} events"):
            dataset = EvalDataset(
                events=cache,
                event_index=index,
                times=times,
                config=config.data,
                grid=grid,
                usable=availability.usable(store.origin_time(index)),
                retrieval=retrieval,
            )
            predictions, neighbours = predict_cutouts(model, dataset, device)
            retrieved_per_event.append(neighbours)

            predicted = first_warning_times(predictions, times, store.pga_thresholds, alpha)
            actual = true_exceedance_times(store, handle, cache, grid, index, len(store.pga_thresholds))
            epicentre = store.metadata.iloc[index][coord_keys].values.astype(float)
            results.append((predicted, actual, hypocentral_distances(grid, epicentre)))

    with open(output_dir / f"{args.checkpoint}_warning.pkl", "wb") as f:
        pickle.dump((list(times), [], results, alpha), f)
    np.save(output_dir / "retrieved_events", np.stack(retrieved_per_event))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
