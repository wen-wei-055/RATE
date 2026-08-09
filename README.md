# RATE: A Retrieval-Augmented Transformer for Regional Earthquake Early Warning

This repository is the official implementation of **RATE** (Retrieval-Augmented Transformer for Earthquake), a framework designed to enhance real-time earthquake intensity prediction by leveraging historical seismic records. Published in *IEEE Geoscience and Remote Sensing Letters*, vol. 22, pp. 1–5, 2025.

[![Paper: IEEE GRSL](https://img.shields.io/badge/Paper-IEEE%20GRSL-blue)](https://ieeexplore.ieee.org/document/11124201)

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
* **Goal**: Establish foundational weights for the CNN and Transformer modules using large-scale datasets.

### Stage 2: Retrieval Augmentation Training
* **Action**: Link the Stage 1 output by setting the `transfer_model_path` in your configuration.
* **Goal**: Train the model to align current seismic waveforms with historical spatial patterns.
* **Mechanism**: The current event and retrieved historical waveforms are concatenated and passed through the model to refine PGA distribution predictions.

---

## Data Preparation & Setup

### 1. Dataset Generation (TEAM Methodology)
Our data preprocessing pipeline follows the standards established by the **TEAM** (Transformer Earthquake Alerting Model) paper.
* **Format Conversion**: Use `japan.py` (consistent with the [TEAM implementation](https://github.com/yetinam/TEAM)) to convert raw seismic waveforms into **HDF5** format.
* **Data Splitting**: Partition the dataset into **train**, **val**, and **test** sets (e.g., using a 60:10:30 ratio) following the same protocols for fair comparison.

### 2. Metadata & Database Initialization
* **Generate `station.json`**: Define station coordinates and IDs used for positional embedding.
* **Generate Historical Database**: Execute `preprocess.gen_historical_database` to compile the database required for the retrieval stage.

### 3. Station Temporal Constraints
* **Generate `appearance_time`**: Records the **activation (start) time** and **deactivation (end) time** of each station within the dataset.
---

## Experimental Configuration
* **Input Duration**: 30 seconds (3,000 time steps at 100Hz).
* **Optimization**: Adam optimizer with `ReduceLROnPlateau` scheduler.
* **Efficiency**: Average retrieval time is approximately **0.000969 seconds** per query using FAISS, making it feasible for real-time deployment.

---

## Citation

If you use this code or research in your work, please cite:

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
