"""Typed configuration for RATE.

Every run - pre-training or retrieval-augmented training - is described by one
JSON file that maps onto the dataclasses below.  Unknown keys raise instead of
being silently ignored, which is how a number of settings in the original code
ended up doing nothing at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _check_keys(cls: type, raw: dict, where: str) -> dict:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {unknown}. Known keys: {sorted(known)}")
    return dict(raw)


@dataclass
class ModelConfig:
    """Architecture.  ``stations`` is the size of the fixed station grid."""

    stations: int
    #: Number of retrieved historical events fed alongside the current one.
    #: 0 reproduces stage 1 (pre-training), 1 reproduces stage 2 (RATE).
    retrieved_events: int = 0
    trace_length: int = 3000
    borehole: bool = True
    downsample: int = 5
    mlp_dims: tuple[int, ...] = (500, 500, 500)
    output_mlp_dims: tuple[int, ...] = (150, 100, 50, 30, 10)
    wavelength: tuple[tuple[float, float], ...] = ((0.01, 15), (0.01, 15), (0.01, 10))
    rotation: float | None = None
    rotation_anchor: tuple[float, float] | None = None
    alternative_coords_embedding: bool = False
    transformer_layers: int = 6
    n_heads: int = 10
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02
    ffn_hidden_dim: int = 1000
    pga_mixture: int = 5
    #: Reproduce the published attention, in which the query, key and value
    #: projections are three parameters over one buffer.  See NOTES.md; false
    #: gives standard attention and requires retraining.
    legacy_shared_qkv: bool = False
    #: Reproduce the published stage 1, which multiplied the attention and
    #: feed-forward outputs by the station service mask and took their absolute
    #: value.  Only defined without retrieval.  See NOTES.md.
    legacy_station_mask: bool = False

    @property
    def total_stations(self) -> int:
        """Sequence length seen by the transformer."""
        return self.stations * (1 + self.retrieved_events)

    @property
    def channels(self) -> int:
        return 6 if self.borehole else 3


@dataclass
class RetrievalConfig:
    """FAISS retrieval of a similar historical event."""

    database: str = ""
    #: The retrieved event is drawn uniformly from the top-k neighbours.
    topk: int = 1
    #: How far past the current cutout the retrieved waveform may be revealed.
    peek_seconds: float = 25.0
    #: The training split is also the retrieval pool, so by default an event
    #: retrieves *itself*.  See README, "Known deviations".
    exclude_self: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.database)


@dataclass
class DataConfig:
    """One HDF5 file holding every event, ordered train -> val -> test.

    Waveform shapes are not repeated here: ``trace_length`` and ``channels``
    are taken from the model by :meth:`Config.validate`, so there is one place
    where they are decided.
    """

    data_path: str
    station_json: str
    train_val_boundary: int
    val_test_boundary: int
    first_station_appearance: str | None = None
    last_station_appearance: str | None = None
    event_key: str = "KiK_File"
    magnitude_key: str = "M_J"
    batch_size: int = 5
    sampling_rate: int = 100
    #: Seconds of pre-event noise kept in front of the first P arrival.
    noise_seconds: float = 5
    #: Cutout window relative to the first P arrival, in seconds.
    cutout_start: float = -1
    cutout_end: float = 25
    trigger_based: bool = True
    station_blinding: bool = True
    #: Subtract the mean of the observed part of each trace.  Stage 1 of the
    #: published work did this, stage 2 did not.
    adjust_mean: bool = False
    integrate: bool = False
    min_magnitude: float | None = None
    magnitude_resampling: float = 1.0
    min_upsample_magnitude: float = 5
    upsample_high_station_events: int | None = None
    oversample: int = 1
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def bind(self, model: ModelConfig) -> None:
        self.trace_length = model.trace_length
        self.channels = model.channels

    @property
    def cutout(self) -> tuple[int, int]:
        """Half-open range of sample counts the model is allowed to see."""
        return (
            int(self.sampling_rate * (self.noise_seconds + self.cutout_start)),
            int(self.sampling_rate * (self.noise_seconds + self.cutout_end)),
        )

    @property
    def peek_samples(self) -> int:
        return int(self.sampling_rate * self.retrieval.peek_seconds)


@dataclass
class LegacyTraining:
    """Switches that exist only to reproduce the published runs exactly.

    Each one is a defect that was found during the rewrite and left reachable
    rather than silently fixed; see NOTES.md.  New work should leave them off.
    """

    #: Average the validation loss over every station, including those out of
    #: service, whose target is the -1.5 floor.  Stage 2 did this; stage 1 did
    #: not.  The value drives the scheduler and the checkpoint selection.
    validation_over_all_stations: bool = False
    #: Drop validation batches whose loss is NaN instead of letting it through.
    #: Stage 1 did this; stage 2 did not.
    skip_nan_validation_batches: bool = False


@dataclass
class WeightedLoss:
    """Per-intensity-band weighting of the mixture density loss."""

    #: Off by default: the published runs used the plain mean because the
    #: config key never reached the loss.  See README, "Known deviations".
    enabled: bool = False
    thresholds: tuple[float, ...] = ()
    weights: tuple[float, ...] = ()

    def __post_init__(self):
        if self.enabled and len(self.weights) != len(self.thresholds) + 1:
            raise ValueError("weighted_loss needs one more weight than thresholds")


@dataclass
class TrainingConfig:
    weight_path: str
    epochs: int = 50
    lr: float = 1e-5
    clipnorm: float = 1.0
    device: str = "cuda:0"
    workers: int = 0
    torch_threads: int = 6
    lr_factor: float = 0.8
    lr_patience: int = 4
    #: Checkpoints are kept unconditionally until this epoch, then only when
    #: the validation loss improves on everything since ``keep_all_until``.
    keep_all_until: int = 20
    transfer_model_path: str | None = None
    #: Sub-modules to load-and-freeze, by attribute name on the model.
    freeze: tuple[str, ...] = ()
    weighted_loss: WeightedLoss = field(default_factory=WeightedLoss)
    legacy: LegacyTraining = field(default_factory=LegacyTraining)


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    seed: int = 42

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        raw = _check_keys(cls, raw, "config")
        data_raw = _check_keys(DataConfig, raw.pop("data"), "config.data")
        retrieval = RetrievalConfig(
            **_check_keys(RetrievalConfig, data_raw.pop("retrieval", {}), "config.data.retrieval")
        )
        training_raw = _check_keys(TrainingConfig, raw.pop("training"), "config.training")
        weighted = WeightedLoss(
            **_check_keys(WeightedLoss, training_raw.pop("weighted_loss", {}), "config.training.weighted_loss")
        )
        legacy = LegacyTraining(
            **_check_keys(LegacyTraining, training_raw.pop("legacy", {}), "config.training.legacy")
        )
        config = cls(
            model=ModelConfig(**_check_keys(ModelConfig, raw.pop("model"), "config.model")),
            data=DataConfig(retrieval=retrieval, **data_raw),
            training=TrainingConfig(weighted_loss=weighted, legacy=legacy, **training_raw),
            **raw,
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        def unpack(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {f.name: unpack(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, tuple):
                return [unpack(v) for v in obj]
            return obj

        return unpack(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def validate(self) -> None:
        self.data.bind(self.model)
        if self.model.retrieved_events not in (0, 1):
            raise ValueError("retrieved_events must be 0 (pre-training) or 1 (RATE)")
        if bool(self.model.retrieved_events) != self.data.retrieval.enabled:
            raise ValueError(
                "model.retrieved_events and data.retrieval.database disagree: either "
                "both describe retrieval or neither does"
            )
        if self.model.legacy_station_mask and self.model.retrieved_events:
            raise ValueError(
                "legacy_station_mask only exists for stage 1: the mask covers the current "
                "stations, not the retrieved ones"
            )
        if not 0 < self.data.train_val_boundary <= self.data.val_test_boundary:
            raise ValueError("expected 0 < train_val_boundary <= val_test_boundary")
        start, end = self.data.cutout
        if start <= 0 or end <= start:
            raise ValueError(f"empty cutout range {(start, end)}")
