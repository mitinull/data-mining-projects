# ECG5000 Anomaly Detection

This project detects abnormal ECG heartbeats with autoencoders trained only on normal ECG5000 beats.

## Original contribution

It compares a Conv1D autoencoder and an LSTM autoencoder under the same normal-only split, optimizer, early stopping, and robust reconstruction-error threshold. The threshold is the validation-error median plus three MAD-scaled deviations.

## Setup and run

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/train.py
    jupyter notebook notebooks/01_data_exploration.ipynb

The first training run downloads ECG5000 and writes metrics, reconstruction errors, demo arrays, and model checkpoints to `outputs/`.

## Notebooks

1. `notebooks/01_data_exploration.ipynb` — class imbalance and heartbeat EDA.
2. `notebooks/02_model_results.ipynb` — model metrics, reconstruction errors, and confusion matrices.
3. `notebooks/03_demo_walkthrough.ipynb` — individual-beat reconstructions and anomaly predictions.

Notebooks 2 and 3 include an optional training cell. Skip it when outputs already exist.
