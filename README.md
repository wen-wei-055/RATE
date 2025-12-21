# RATE: A Retrieval-Augmented Transformer for Regional Earthquake Early Warning

This repository is the official implementation of **RATE** (Retrieval-Augmented Transformer for Earthquake), a framework designed to enhance real-time earthquake intensity prediction by leveraging historical seismic records.

[![Paper: IEEE TGRS](https://img.shields.io/badge/Paper-IEEE%20TGRS-blue)](https://ieeexplore.ieee.org/document/11124201)

---

## Methodology Overview

**RATE** is the first framework to integrate a retrieval-augmented (RA) mechanism within a Transformer architecture for earthquake early warning (EEW). 

* **Core Principle**: It integrates ongoing waveform data with similar past events retrieved from a curated database, enabling more accurate and spatially aware predictions.
* **Retrieval Mechanism**: Each event is encoded into a spatial intensity signature. During inference, the system performs a **Cosine Similarity** search using the **FAISS** library to retrieve the most relevant historical event.
* **Architecture**: Combines CNNs for feature extraction and Transformer encoders for multi-station spatial-temporal modeling.



---

## Training Pipeline

The training process follows a specific two-stage workflow to effectively integrate the retrieval-augmented mechanism.

### Stage 1: Pre-training (Base Model)
* **Action**: Modify the parameters in `config.json` to define the model architecture and training hyperparameters.
* **Goal**: Establish foundational weights for the CNN and Transformer modules using large-scale datasets from Japan (KiK-net) or Taiwan (CWASN).

### Stage 2: Retrieval Augmentation Training
* **Action**: Link the Stage 1 output by setting the `transfer_model_path` in your configuration.
* **Goal**: Train the model to align current seismic waveforms with historical spatial patterns.
* **Mechanism**: The current event and retrieved historical waveforms are concatenated and passed through the model to refine PGA distribution predictions.

---

## Data Preparation & Setup

### 1. Format Conversion & Splitting
* **Conversion**: Use `japan.py` (for KiK-net) or appropriate scripts for CWASN to convert raw data into **HDF5** format.
* **Splitting**: Partition data into `train`, `val`, and `test` sets (e.g., Japan dataset uses a 60:10:30 ratio).

### 2. Metadata & Database Initialization
* **Generate `station.json`**: Define station coordinates and IDs used for positional encoding.
* **Generate Historical Database**: Execute `preprocess.gen_historical_database` to compile the database $\Phi$ required for the retrieval stage.

### 3. Station Temporal Constraints
* **Generate `appearance_time`**: Records the operational windows of stations to ensure training data cleanliness.
* > **Warning (Information Leakage):** This file is for **Training** purposes. Avoid including these constraints during the **Evaluation** phase unless station status is explicitly known, to prevent unrealistic performance gains via information leakage.

---

## Experimental Configuration
* **Input Duration**: 30 seconds (3,000 time steps at 100Hz).
* **Optimization**: Adam optimizer with `ReduceLROnPlateau` scheduler.
* **Efficiency**: Average retrieval time is approximately **0.000969 seconds** per query using FAISS, making it feasible for real-time deployment.

---

## Citation

If you use this code or research in your work, please cite:

```bibtex
@article{lin2025rate,
  title={RATE: A Retrieval-Augmented Transformer for Regional Earthquake Early Warning},
  author={Lin, Wen-Wei and Chen, Kuan-Yu and Chen, Da-Yi},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}
}