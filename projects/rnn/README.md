# Multivariate Weather Forecasting

This project forecasts Jena Climate temperature (`T (degC)`) and relative humidity (`rh (%)`) from a 72-hour history of all 14 weather sensors. It compares direct LSTM, direct GRU, sequence-to-sequence LSTM, and attention sequence-to-sequence LSTM forecasts at 6-, 24-, and 72-hour horizons.

## Original contribution

The contribution is a controlled temporal-attention ablation. The hypothesis is that attention over encoder states improves the 72-hour forecast more than the 6-hour forecast, because it can retrieve earlier daily and weather-regime information rather than relying only on the final recurrent state.

## Setup and run

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/train.py
    python scripts/build_report_assets.py
    jupyter notebook notebooks/01_data_exploration.ipynb

The first run downloads the official TensorFlow/Keras Jena Climate archive, downsamples it to hourly readings, and saves real metrics, predictions, histories, and checkpoints under `outputs/`. Forecast origins are six hours apart to avoid near-duplicate overlapping windows. The default run trains all four models; use `--epochs 1 --patience 1` only for a smoke test.

## Notebooks and report

1. `notebooks/01_data_exploration.ipynb` explores the real data and windowing choices.
2. `notebooks/02_model_results.ipynb` compares the models and horizons.
3. `notebooks/03_demo_walkthrough.ipynb` shows chronological forecasts from all four RNN models.

Notebooks 2 and 3 contain an optional training cell. Skip it if `outputs/results.csv` already exists. Build the report after artifacts are available:

    cd report
    pdflatex -interaction=nonstopmode report.tex
