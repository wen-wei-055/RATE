# RATE: A Retrieval-Augmented Transformer for Regional Earthquake Early Warning

This repository is the official implementation of **RATE** (Retrieval-Augmented Transformer for
Earthquake), a framework designed to enhance real-time earthquake intensity prediction by
leveraging historical seismic records. Published in *IEEE Geoscience and Remote Sensing Letters*,
vol. 22, pp. 1–5, 2025.

[![Paper: IEEE GRSL](https://img.shields.io/badge/Paper-IEEE%20GRSL-blue)](https://ieeexplore.ieee.org/document/11124201)

---

## Method

RATE is the first framework to integrate a retrieval-augmented mechanism within a transformer
architecture for earthquake early warning.

* **Core principle** — the waveforms arriving right now are read together with a similar past
  event, so the model can lean on how that event went on to develop.
* **Retrieval** — every event is encoded, once per elapsed second, as the vector of peak
  accelerations observed so far at each station: its spatial intensity signature. At inference the
  most similar past event is found by cosine similarity over those vectors, with FAISS
  (~0.001 s per query).
* **Architecture** — a CNN embeds each station's waveform, a transformer mixes information across
  stations, and every station is decoded into a mixture of Gaussians over its log10 PGA. A
  retrieved event simply extends the station axis: the transformer sees `2 × stations` slots and
  attends across both.

## Layout

```
rate/            the implementation - one pipeline, not one per training stage
  config.py      typed configuration; unknown keys are rejected
  stations.py    the fixed station grid and when each station was in service
  events.py      access to the HDF5 event archive
  retrieval.py   the FAISS index over spatial intensity signatures
  data.py        batch construction, with or without a retrieved event
  model.py       the network
  losses.py      mixture density loss
  train.py       training entry point (both stages)
  evaluate.py    warning-time evaluation
  preprocess.py  builds the retrieval database
configs/         one JSON per training stage
dataset_configs/ example station tables and retrieval database
tests/           a synthetic dataset and an end-to-end smoke test
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

One HDF5 file holds every event, ordered `[train | val | test]`, with the two boundaries given in
the config. Use `japan.py` from the [TEAM implementation](https://github.com/yetinam/TEAM) to
convert raw waveforms into it; the published work used a 60:10:30 split.

```
metadata/event_metadata     one row per event: event name, magnitude, hypocentre
metadata/sampling_rate      Hz
metadata/pga_thresholds     intensity levels warnings are issued for, in g
metadata/time_before        seconds of noise kept in front of the first P arrival
data/<event>/waveforms      (stations, samples, channels)
data/<event>/coords         (stations, 4)  lat, lon, depth, borehole depth
data/<event>/p_picks        (stations,)    sample index of the P arrival
data/<event>/pga            (stations,)    log10 peak ground acceleration
data/<event>/pga_times      (stations, thresholds)
```

The event name must begin with a 14-character timestamp (`YYYYMMDDhhmmss`); that is how an event
is matched against each station's service window.

Two more files describe the network itself, both keyed by `"lat,lon,depth,borehole_depth"`:
`station_json` maps every station to its row in the grid, and the appearance-time files give the
date each station entered and left service. Examples are in `dataset_configs/`.

## Running

```bash
# stage 1 - pre-training, no retrieval
python -m rate.train --config configs/pretrain.json

# the retrieval database, built from the same archive
python -m rate.preprocess --config configs/rate.json

# stage 2 - retrieval-augmented training, starting from the stage 1 weights
python -m rate.train --config configs/rate.json

# warning times on the test split
python -m rate.evaluate --experiment-path weight_path/rate --checkpoint 45
```

Add `--test-run` to training for a ten-event dry run. `python -m tests.smoke_test` runs the whole
pipeline on a synthetic dataset in about a minute, without any real data.

Training writes to `weight_path`: `config.json` (the exact configuration, which
`rate.evaluate` reads back), a copy of the code that produced the run, `checkpoint_NN.pth`,
`metrics.json`, `train.log` and a confusion matrix per split.

## Configuration

The two stages differ only in their config. The keys that decide which is which:

| key | stage 1 | stage 2 |
| --- | --- | --- |
| `model.retrieved_events` | `0` | `1` |
| `data.retrieval.database` | absent | path to the `.npy` from `rate.preprocess` |
| `training.transfer_model_path` | absent | the stage 1 checkpoint |
| `training.freeze` | `[]` | `["TotalEmbedding", "To_GaussianDistribution"]` |

Everything else is documented in `rate/config.py`, which is the only place a config key exists.
A key that is not listed there is a typo and raises rather than being ignored.

Notable settings:

* `data.cutout_start` / `cutout_end` — how much of the record the model may see, in seconds
  relative to the first P arrival. One value is drawn per batch.
* `data.retrieval.peek_seconds` — how far past that cutout the *retrieved* event may be revealed.
* `data.station_blinding` — drop a random subset of stations per batch, as augmentation.
* `data.trigger_based` — hide stations that have not triggered yet, so no future leaks in.
* `model.legacy_*`, `training.legacy.*`, `data.retrieval.exclude_self`, `data.adjust_mean` — see
  below.

## Reproducing the published models

The shipped configs reproduce the published runs exactly: given the same archive and seed, this
code writes the same weights, the same optimiser state and the same losses as the code it
replaces, verified by running both. Existing checkpoints load and evaluate unchanged.

That is what the `legacy` switches are for. Each one preserves a behaviour that turned out to be
a defect, rather than fixing it silently. `NOTES.md` explains each in full; in short:

* `model.legacy_shared_qkv: true` — the query, key and value projections are three parameters
  over one buffer, so they are equal for the whole run. `false` gives standard attention.
* `model.legacy_station_mask: true` (stage 1 only) — the attention and feed-forward outputs are
  multiplied by the station service mask and passed through `abs`.
* `training.legacy.validation_over_all_stations` — stage 2 averaged its validation loss over
  out-of-service stations too, and that number drove the scheduler; stage 1 did not.
* `training.legacy.skip_nan_validation_batches` — stage 1 dropped NaN validation batches;
  stage 2 let them through.
* `data.retrieval.exclude_self: false` — the retrieval pool is the training split, so during
  stage 2 training each event retrieves *itself*.
* `data.adjust_mean` — stage 1 removed the mean of the observed part of each trace, stage 2 did
  not, and stage 2 freezes the embedding stage 1 trained.

Turning any of them off is a research decision that means retraining. Evaluation has one of its
own: `--ignore-station-windows` reproduces a run of the old `evaluate.py` that was not given
`--first_station_appearance_path` / `--last_station_appearance_path`, which was the documented
way to call it.

## Citation

```bibtex
@ARTICLE{11124201,
  author={Lin, Wen-Wei and Chen, Kuan-Yu and Chen, Da-Yi},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={RATE: A Retrieval-Augmented Transformer for Regional Earthquake Early Warning},
  year={2025},
  volume={22},
  number={},
  pages={1-5},
  keywords={Earthquakes;Accuracy;Transformers;Training;Real-time systems;Adaptation models;Electronics packaging;Data models;Context modeling;Predictive models;Deep learning;earthquake early warning (EEW);regional warning systems;retrieval augmented (RA);seismic intensity prediction},
  doi={10.1109/LGRS.2025.3598322}}
```
