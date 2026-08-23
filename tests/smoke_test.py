"""End-to-end run on the synthetic dataset.

    python -m tests.smoke_test

Builds a small archive, runs both training stages and the evaluation, and
checks the artefacts they are supposed to leave behind.  It takes about a
minute on a laptop CPU and needs no real data.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np

from rate import evaluate, preprocess, train
from tests import synthetic


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        paths = synthetic.write(directory)
        configs = synthetic.configs(directory, paths)

        print("== stage 1: pre-training, no retrieval")
        train.main(["--config", str(configs["pretrain"])])
        assert (directory / "pretrain" / "checkpoint_01.pth").is_file()

        print("== retrieval database")
        preprocess.main(["--config", str(configs["rate"]), "--time-steps", "6"])
        signatures = np.load(directory / "signatures.npy")
        assert signatures.shape == (synthetic.N_EVENTS, synthetic.N_STATIONS, 7), signatures.shape
        lengths = np.sqrt((signatures[:, :, 1:] ** 2).sum(axis=1))
        assert np.allclose(lengths, 1, atol=1e-5), lengths

        print("== stage 2: retrieval-augmented training, transferred from stage 1")
        train.main(["--config", str(configs["rate"])])
        assert (directory / "rate" / "checkpoint_01.pth").is_file()

        print("== evaluation")
        evaluate.main([
            "--experiment-path", str(directory / "rate"),
            "--checkpoint", "01",
            "--blind-time", "0.5",
            "--last-time", "2.5",
            "--time-step", "0.5",
        ])
        output = directory / "rate" / "evaluation" / "test"
        with open(output / "01_warning.pkl", "rb") as f:
            times, _, results, alpha = pickle.load(f)
        assert len(results) == synthetic.N_EVENTS - 9
        predicted, actual, distance = results[0]
        assert predicted.shape == (synthetic.N_STATIONS, len(synthetic.THRESHOLDS), len(alpha))
        assert actual.shape == (synthetic.N_STATIONS, len(synthetic.THRESHOLDS))
        assert distance.shape == (synthetic.N_STATIONS,) and np.all(distance > 0)
        print("== the same evaluation without retrieval, on the validation split")
        evaluate.main([
            "--experiment-path", str(directory / "pretrain"),
            "--checkpoint", "01", "--val",
            "--blind-time", "0.5", "--last-time", "2.5", "--time-step", "0.5",
        ])
        assert (directory / "pretrain" / "evaluation" / "val" / "01_warning.pkl").is_file()

        retrieved = np.load(output / "retrieved_events.npy")
        assert retrieved.shape == (3, len(times), 10), retrieved.shape
        assert np.all(retrieved[retrieved >= 0] < 6), "retrieval must stay inside the training split"

        print("\nall good")


if __name__ == "__main__":
    main()
