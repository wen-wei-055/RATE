# Rewrite notes

This is the record of a cleanup pass over the original code: what was merged, what was found
along the way, and what changed as a result. It is written for whoever maintains the repository,
not for a reader of the paper.

## Why there is only one pipeline now

`Pretrain/` and `RATE/` were two copies of the same five files. Diffing them, the *behavioural*
difference came down to:

| | `Pretrain/` | `RATE/` |
| --- | --- | --- |
| retrieved event | none | one, appended along the station axis |
| transformer length | `stations` | `2 × stations` |
| starting weights | random | the stage 1 checkpoint |
| frozen sub-modules | none | embedding and output head |
| mean removal | on | off |

Everything else was whitespace, reordered arguments and comments deleted on one side. All five
of those rows are configuration, not code, so the two directories collapsed into one package
where stage 1 is `retrieved_events: 0` and stage 2 is `retrieved_events: 1`. Nothing about the
two-*stage* training procedure changed - only the two *copies* are gone.

## Findings that change what the numbers mean

These were found by running the original code, not by reading it. **Every one of them is
preserved**, behind a named switch that the shipped configs set to what the published runs did, so
the rewrite reproduces the original bit for bit. Each is also a decision worth making
deliberately, which is why they are switches and not silent behaviour.

### 1. Query, key and value were one shared matrix

```python
initialization_1 = torch.distributions.Uniform(...).sample((d_model, d_key * n_heads))
self.WQ = nn.Parameter(initialization_1)
self.WK = nn.Parameter(initialization_1)   # same storage, not a copy
self.WV = nn.Parameter(initialization_1)
```

`nn.Parameter(t)` wraps `t` without copying it, so `WQ`, `WK` and `WV` are three parameters
pointing at one buffer. They start equal and every in-place optimiser update writes through all
three, so they stay equal for the whole run: the attention is `softmax(xW (xW)ᵀ) xW`. Confirmed
directly - `WQ.data_ptr() == WK.data_ptr() == WV.data_ptr()`, still equal after an Adam step.

`model.legacy_shared_qkv: true` reproduces this exactly, aliasing included. It is worth being
precise about why: the obvious clean equivalent - one parameter under three names - is *not* the
same model. With three parameters, autograd keeps three gradients and Adam keeps three moment
estimates, and each takes its own normalised step through the shared buffer; with one parameter
there is a single gradient (their sum) and a single step. The published dynamics need the
aliasing, so the aliasing is what the switch reproduces.

`false` gives standard attention with independent projections, which requires retraining.

### 2. During stage 2 training, each event retrieves itself

The FAISS pool is `signatures[:train_val_boundary]`, i.e. the training split, and training
queries are training events. An event's own signature is in the index and is its own nearest
neighbour at cosine 1, so with `topk: 1` the "historical" block is the same event - revealed
25 seconds (`peek_seconds`) further into the future than the current block, which is exactly the
interval the target PGA is drawn from. Confirmed: every training event retrieves itself.

Validation and test events are not in the pool, so they retrieve a genuinely different event.
Training therefore rewards reading the answer off a block that at evaluation time contains
something else. `data.retrieval.exclude_self: true` skips the self match; the shipped config
leaves it `false` because that is what the published runs did.

### 3. Stage 1 and stage 2 fed the embedding differently

`Pretrain/util.py` subtracted the mean of the observed part of every trace
(`adjust_mean`, defaulted to `True` and never set in the config). `RATE/util.py` dropped that
code path entirely. Stage 2 then *freezes* the waveform embedding - so the frozen encoder is fed
a distribution it never saw during pre-training. Both behaviours are preserved in the two shipped
configs (`data.adjust_mean: true` / `false`); making them agree means retraining.

### 4. Stage 1 applied `torch.abs` to the attention output

```python
o = torch.matmul(o, self.WO)
o = torch.abs(o * station_mask)     # Pretrain only
```

The mask zeroes stations that were out of service, which is harmless, but `abs` is applied to
every station and forces the whole residual branch non-negative. The feed-forward output was
masked the same way. Stage 2 has neither line.

`model.legacy_station_mask: true` (stage 1's config) reproduces both, and is rejected together
with retrieval, since the mask only covers the current stations. Turning it off changes what
pre-training optimises and means re-running stage 1.

### 5. The weighted loss never reached the loss

The config defines `"weighted_loss": {threshold, weight}`; `train.py` read
`training_params.get('weighted_loss_threshold')`, a key that does not exist, so the thresholds
were always `None`. Even had the key matched, `time_distributed_loss` guarded the weighted branch
with `if weight is True`, and `weight` is a list. Both published stages trained on the plain mean.

The rewrite implements what the code was reaching for - `sum(wᵢlᵢ) / sum(wᵢ)` - behind
`training.weighted_loss.enabled`, which the shipped configs leave `false`.

### 6. Validation loss disagreed with training loss (stage 2)

Stage 2's training loop dropped out-of-service stations before computing the loss; its validation
loop dropped them *after*, so the validation loss averaged in ~700 stations whose target had been
replaced by the `-1.5` floor. That value drove `ReduceLROnPlateau` and the checkpoint selection.
Stage 1 masked correctly in both loops.

`training.legacy.validation_over_all_stations` keeps the difference: `true` in the RATE config,
`false` in the pre-training one, exactly as the two originals behaved. The confusion matrix was
always built from the masked stations, in both.

### 7. Stage 1 dropped validation batches whose loss was NaN

`Pretrain/train.py` had `if torch.isnan(val_loss): continue`, which also skipped the batch's
contribution to the confusion matrix while still counting it in the denominator of the epoch
average. Stage 2 had no such guard, so a single NaN poisoned the epoch, the scheduler and the
checkpoint selection. Preserved per stage as `training.legacy.skip_nan_validation_batches`.

### 8. Evaluation ignored the station service windows unless asked

`evaluate.py` took the appearance-time files from the command line rather than from the config,
and the README never mentioned the flags. Left off - the normal case - every station counts as in
service at evaluation, while training excluded them. The rewrite reads them from the config the
run saved, so the two agree; `--ignore-station-windows` reproduces an evaluation run that did not
pass the flags. It only affects stations that were out of service, whose true exceedance times
are NaN anyway.

## Things that were simply broken, and are gone

* `TrainDevTestSplitter` dispatched to `TrainDevTestSplitter.test_2016` and `.test_2011`, neither
  of which exists; any call would have raised `AttributeError`. Nothing called it - splitting is
  done when the archive is built.
* `transfer_weights(model, ...)` copied into `own_state` and then called
  `full_model.load_state_dict(...)` - the caller's global, not its argument. It worked only
  because the two happened to be the same object.
* `gaussian_confusion_matrix` read `training_params` and `epoch` out of module globals.
* Hard-coded shapes that made every other configuration silently wrong: `707` stations,
  `3000` samples, `6` channels, `500` embedding dimensions, `31` retrieval time steps, and `865`
  as the width of the convolution stack. All are derived now - which is why the smoke test can
  run a 8-station, 20-dimensional model that the original code could not build.
* `preprocess/gen_historical_database.py` built station keys from three coordinates while the
  station tables are keyed by four, so it raised `KeyError` on the example files. It also
  re-normalised the entire array once per event, from inside the event loop.
* The "no neighbour found" fallbacks allocated 3-channel placeholders for 6-channel data and
  referenced an undefined `pair_pga`; both would have raised had FAISS ever returned `-1`.
* Filtering `event_metadata` by `min_mag` renumbered the rows the retrieval index and the total
  archive are addressed by, silently pairing events with the wrong data. Filtering now keeps
  archive row indices intact.
* `evaluate.py` did `coords_event[2]` on a Series indexed by column name, which pandas has since
  stopped accepting: the original evaluation raises `KeyError: 2` on any recent pandas. The
  rewrite reads the hypocentre as an array.
* Roughly thirty dead parameters, flags and locals: `--experiment_retrieve_event` and its
  `test_retrieve_event` config key (passed into `**kwargs` and dropped), `--n_pga_targets`,
  `--dataset_id`, `--loss_limit`, `--times`, `--no_multiprocessing`, `--additional_data` (which
  called `load_events` with the wrong signature), `max_ensemble_size`, `pga_mode`,
  `sliding_window`, `stations_channel_class`, `stations_channel_boolean`, `latlon_IDtable`,
  `coords_target`, `select_first`, `translate`, `p_pick_limit`, `reverse_index`, `EARTH_RADIUS`,
  the seaborn import, the empty `stats.json`, and the ensemble machinery around a single model.

## Other behaviour changes

* The train/val/test archives are gone; there is one `data_path` plus the two boundaries. The old
  code already required the total archive to be the three splits concatenated in order - it read
  metadata from the split file and waveforms from the total file at the same index - so this
  removes a way to get that wrong, not a feature.
* Both confusion matrices now use the thresholds stored in the dataset (`pga_thresholds`) rather
  than the list hard coded in `train.py`, and both use g = 9.81; stage 2's validation used 9.8.
  This is the one place where a number can still differ from the original, and only in the
  confusion matrix text file: a station sitting within 0.04% of a threshold could land in a
  different cell. It reaches no weight, no loss and no evaluation output, and in the parity runs
  below the matrices came out identical. If your archive's `pga_thresholds` are the
  `[0.01, 0.02, 0.05, 0.1, 0.2]` the old `train.py` assumed - the same list `evaluate.py` has
  always read from the file - the thresholds themselves are unchanged.
* The evaluation cutout was `sampling_rate * (time + 5)` with the 5 hard coded, next to a
  `noise_seconds` setting that training read from the config. It now reads the same setting, and
  warns when the archive's own `time_before` disagrees with it. Both are 5 in the shipped configs,
  so nothing moves.
* Evaluation used to read every event of the archive into memory before starting, because the
  retrieval step needs random access to any training event. It now reads from the open file as it
  goes, which is what training already did.
* Station service windows are parsed once instead of once per event per station (that was
  `n_events × n_stations` `UTCDateTime` constructions per data load).
* An event listed in the metadata but missing from `data/` now raises. The old code counted it as
  "skipped" but had already appended its service mask, shifting everything after it.
* Output file names: `metrics.txt` (a `str(dict)` dump) is now `metrics.json`;
  `total_choose_event_list.npy` is now `retrieved_events.npy`. In the warning pickle, the first
  element is now the times the model was actually evaluated at - it used to be the unrelated
  `--times` flag, which nothing else used.
* `evaluate.py` no longer takes `--first_station_appearance_path` / `--last_station_appearance_path`
  / `--experiment_path`-relative weight names; it reads the config the run saved and takes
  `--experiment-path` and `--checkpoint`.

## What was verified identical

On a synthetic archive, against the original code checked out from this repository's history.
The end-to-end runs are the ones that matter: the original `train.py` and `python -m rate.train`
were pointed at the same data with the same config, and the checkpoints they wrote were compared
tensor by tensor.

| | stage 1 (`Pretrain/`) | stage 2 (`RATE/`) |
| --- | --- | --- |
| weights after a full epoch | identical | identical |
| Adam moment estimates | identical | identical |
| per-epoch train and validation losses | identical | identical |
| confusion matrix files | identical | identical |

Component level, against `RATE/`:

| | result |
| --- | --- |
| `state_dict` keys and shapes | identical |
| initial weights from the same seed | identical |
| forward pass, same weights and inputs | identical |
| mixture density loss | identical |
| gradients of every parameter | identical |
| training batches - waveforms, coordinates, PGA targets, service mask | identical, including retrieval, station blinding and trigger masking |
| oversampling: which events an epoch draws, at four settings | identical |
| evaluation: warning times, true exceedance times, distances, retrieved events | identical |

The evaluation comparison is the end-to-end one: the original `evaluate.py` and
`python -m rate.evaluate` were run on the same checkpoint, and the warning times matched entry
for entry - with 774 of 1080 entries being actual warnings rather than NaN, so the comparison is
looking at something.

Two rewrites had to be undone to get there, both invisible in the forward pass and both showing
up only in the gradients:

* `LayerNorm` computed `gamma * (x - m) / sqrt(s + eps)` instead of the original's
  `gamma * ((x - m) / sqrt(s + eps))`. Same value, different rounding, and the gradient with
  respect to `gamma` differed in the last bits.
* Query, key and value had been tied cleanly to one parameter rather than aliased (see finding 1).

Both are worth remembering: an expression that is algebraically equal is not numerically equal,
and "the same weights" is not the same thing as "the same optimiser state".

The original was checked against itself first, to confirm the machine reproduces runs bit for bit
at all; it does, so a difference of 7e-9 was a real difference and not noise.

The original code these runs used is `git show 42e37d4:RATE/train.py` and friends. Two edits were
needed to get it to run at all on a current environment, neither of which touches a number: the
`707` hard-coded in `util.py` had to become `len(stations_table)` for a fixture with a different
station count, and `coords_event[2]` in `evaluate.py` had to become `coords_event.iloc[2]` for
pandas.

Checkpoints from the published runs load and evaluate unchanged.

## Config migration

| old | new |
| --- | --- |
| `model_params.current_station` | `model.stations` |
| `model_params.historical_station` (`null` / `707`) | `model.retrieved_events` (`0` / `1`) |
| `model_params.mad_params.*`, `ffn_params.hidden_dim` | `model.n_heads`, `attention_dropout`, `initializer_range`, `ffn_hidden_dim` |
| `training_params.total_data_path` + `train/val/test_data_path` | `data.data_path` |
| `training_params.station_json_file` | `data.station_json` |
| `training_params.generator_params[0].*` | `data.*` |
| `generator_params[0].historical_database_path`, `RAG_topK`, `peek_sample` | `data.retrieval.database`, `topk`, `peek_seconds` |
| `training_params.epochs_full_model` | `training.epochs` |
| `training_params.weighted_loss` | `training.weighted_loss` (with `enabled`) |
| `epochs_single_station`, `filter_single_station_by_pick` | removed, never read |
| `model_params.tie_qkv` (never existed) | `model.legacy_shared_qkv`, `model.legacy_station_mask`, `training.legacy.*` - the switches above |

## Reproducing a published run

The two shipped configs are already set to reproduce their stage. Point `data.data_path` at your
archive, set the two boundaries, and check the one assumption both the old and the new code make
about it - that the archive is the three splits concatenated, in order:

```python
import pandas as pd
key = 'KiK_File'
total = pd.read_hdf('japan.hdf5', 'metadata/event_metadata')[key]
train = pd.read_hdf('train_japan.hdf5', 'metadata/event_metadata')[key]
val   = pd.read_hdf('val_japan.hdf5', 'metadata/event_metadata')[key]
assert list(total[:8107]) == list(train)          # train_val_boundary
assert list(total[8107:9458]) == list(val)        # val_test_boundary
```

The old code read an event's metadata from the split file and its waveforms from the total file
at the same index, so it already depended on this; it just never said so.
