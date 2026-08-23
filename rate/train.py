"""Training entry point.

    python -m rate.train --config configs/pretrain.json     # stage 1
    python -m rate.train --config configs/rate.json         # stage 2

The two stages differ only in their config: stage 2 retrieves a historical
event, starts from the stage 1 weights and freezes the embedding and the
output head.  There is no second copy of the code.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import TrainDataset, identity_collate
from .events import EventStore
from .losses import pga_loss
from .model import FullModel
from .retrieval import RetrievalIndex
from .stations import StationAvailability, StationGrid, load_time_table

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#: Target used for stations that reported no peak acceleration, log10 m/s^2.
PGA_FLOOR = -1.5
#: Standard gravity, for turning threshold levels in g into log10 m/s^2.
G = 9.81


class ExceedanceMatrix:
    """Counts predicted versus observed intensity levels.

    A warning is issued for a station when the predicted probability of
    exceeding a threshold passes ``exceedance_prob``; the highest threshold
    warned for is compared against the highest threshold actually exceeded.
    """

    def __init__(self, thresholds: np.ndarray, exceedance_prob: float = 0.2):
        self.thresholds = thresholds
        self.exceedance_prob = exceedance_prob
        self.counts = np.zeros((len(thresholds) + 1, len(thresholds) + 1), dtype=np.int32)

    def accumulate(self, targets: np.ndarray, pred: np.ndarray) -> None:
        from scipy.stats import norm

        observed = np.sum(targets.reshape(-1, 1) >= self.thresholds, axis=1)
        warned = np.zeros((pred.shape[0], len(self.thresholds)), dtype=int)
        for j, level in enumerate(self.thresholds):
            probability = np.sum(
                pred[:, :, 0] * (1 - norm.cdf((level - pred[:, :, 1]) / pred[:, :, 2])), axis=-1
            )
            warned[:, j] = probability >= self.exceedance_prob
        warned = np.sum(warned, axis=1)
        for predicted, actual in zip(warned, observed):
            self.counts[predicted][actual] += 1

    def write(self, path: Path, epoch: int, loss: float, lr: float) -> None:
        with open(path, "a+", encoding="utf-8") as f:
            f.write(f"{epoch}\n{self.counts}\n\nloss: {loss},   lr: {lr}\n\n")


def build_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="a")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def transfer_weights(model: FullModel, weights_path: str) -> None:
    """Load whatever a previous stage's checkpoint has in common with ``model``."""
    path = Path(weights_path)
    if path.is_dir():
        checkpoints = sorted(p for p in path.iterdir() if p.name.startswith("checkpoint_"))
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoint_* in {path}")
        path = checkpoints[-1]
    print(f"Transferring weights from {path}")
    state = torch.load(path, map_location="cpu")["model_weights"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    for name in missing:
        print(f"  not in the checkpoint, keeping the initial value: {name}")
    for name in unexpected:
        print(f"  in the checkpoint but not in this model, ignored: {name}")


def freeze(model: FullModel, names) -> list[torch.nn.Module]:
    """Freeze whole sub-modules by attribute name, e.g. ``TotalEmbedding``."""
    frozen = []
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            raise AttributeError(f"cannot freeze {name!r}: the model has no such sub-module")
        for parameter in module.parameters():
            parameter.requires_grad = False
        frozen.append(module)
    return frozen


def usable_rows(pred, targets, usable):
    """Flatten a batch to one row per station, dropping stations out of service."""
    pred = pred.contiguous().view(-1, pred.shape[-2], pred.shape[-1])
    targets = targets.contiguous().view(-1)
    keep = torch.nonzero(usable.contiguous().view(-1) == 1, as_tuple=False).squeeze(dim=1)
    return pred[keep], targets[keep]


def run_epoch(model, batches, device, config, matrix, optimizer=None, frozen=(), description=""):
    """One pass over the data; training if an optimizer is given."""
    training = optimizer is not None
    model.train(training)
    for module in frozen:
        module.eval()

    loop = tqdm(batches, desc=description)
    total = 0.0
    with torch.set_grad_enabled(training):
        for (waveforms, coords), (targets, usable) in loop:
            waveforms, coords = waveforms.to(device), coords.to(device)
            targets, usable = targets.to(device), usable.to(device)
            targets[targets == 0] = PGA_FLOOR

            pred = model.predict_current(waveforms, coords)
            pred, targets = usable_rows(pred, targets, usable)
            loss = pga_loss(targets, pred, config.training.weighted_loss)

            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), config.training.clipnorm)
                optimizer.step()
                optimizer.zero_grad()

            total += loss.item()
            loop.set_postfix(loss=loss.item())
            matrix.accumulate(
                targets.detach().cpu().numpy(), pred.detach().cpu().numpy()
            )
    return total / max(len(loop), 1)


def keep_checkpoint(epoch: int, val_losses: list[float], keep_all_until: int) -> bool:
    """Keep every early checkpoint, then only those that improve on the rest."""
    if epoch % 9 == 0 or keep_all_until <= epoch < keep_all_until + 11:
        return True
    if epoch < keep_all_until:
        return False
    history = val_losses[keep_all_until - 1 : -1]
    return bool(history) and val_losses[-1] < min(history)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test-run", action="store_true", help="a few events only, to check the plumbing")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.set_num_threads(config.training.torch_threads)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")

    weight_path = Path(config.training.weight_path)
    weight_path.mkdir(parents=True, exist_ok=True)
    config.save(weight_path / "config.json")
    shutil.copytree(
        Path(__file__).parent,
        weight_path / "exec_code",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    grid = StationGrid.load(config.data.station_json)
    availability = StationAvailability(
        grid,
        load_time_table(config.data.first_station_appearance),
        load_time_table(config.data.last_station_appearance),
    )
    store = EventStore(config.data.data_path, config.data.event_key, config.model.trace_length)
    if len(grid) != config.model.stations:
        raise ValueError(
            f"{config.data.station_json} has {len(grid)} stations, "
            f"model.stations is {config.model.stations}"
        )
    retrieval = RetrievalIndex.build(config.data.retrieval, pool=config.data.train_val_boundary)

    limit = 10 if args.test_run else None
    splits = {
        split: TrainDataset(store, split, config.data, availability, grid, retrieval, limit)
        for split in ("train", "val")
    }
    loaders = {
        split: DataLoader(
            dataset,
            shuffle=(split == "train"),
            batch_size=None,
            collate_fn=identity_collate,
            num_workers=config.training.workers,
        )
        for split, dataset in splits.items()
    }

    model = FullModel(config.model).to(device)
    if config.training.transfer_model_path:
        transfer_weights(model, config.training.transfer_model_path)
    frozen = freeze(model, config.training.freeze)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.training.lr_factor, patience=config.training.lr_patience
    )
    thresholds = np.log10(np.asarray(store.pga_thresholds) * G)

    logger = build_logger(weight_path / "train.log")
    logger.info("start training on %s, %d train / %d val batches",
                device, len(splits["train"]), len(splits["val"]))
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(config.training.epochs):
        for split in ("train", "val"):
            matrix = ExceedanceMatrix(thresholds)
            loss = run_epoch(
                model,
                loaders[split],
                device,
                config,
                matrix,
                optimizer=optimizer if split == "train" else None,
                frozen=frozen,
                description=f"[{split} epoch {epoch + 1}/{config.training.epochs}]",
            )
            matrix.write(
                weight_path / f"{split}_confusion_matrix.txt",
                epoch,
                loss,
                optimizer.param_groups[0]["lr"],
            )
            history[f"{split}_loss"].append(loss)
            logger.info("[%s] epoch: %d -> loss: %.4f", split, epoch, loss)

        scheduler.step(history["val_loss"][-1])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if keep_checkpoint(epoch, history["val_loss"], config.training.keep_all_until):
            with open(weight_path / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
            print("-----Saving checkpoint-----")
            torch.save(
                {
                    "model_weights": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                weight_path / f"checkpoint_{epoch:02d}.pth",
            )


if __name__ == "__main__":
    main()
