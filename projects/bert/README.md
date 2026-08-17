# BERT Text Classification

This project fine-tunes DistilBERT on AG News.
It compares DistilBERT with a TF-IDF baseline and tests data efficiency.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/train.py
jupyter notebook notebooks/01_data_exploration.ipynb
```

The default run uses at most 3,000 training rows, 1,000 test rows, and one epoch.
The first run downloads AG News and the DistilBERT model.

## Notebooks

Run these from the `projects/bert` directory, in order:

1. `notebooks/01_data_exploration.ipynb` explores AG News labels, article lengths, and class vocabulary.
2. `notebooks/02_model_results.ipynb` compares the four saved experiment runs.
3. `notebooks/03_demo_walkthrough.ipynb` predicts several example sentences with all four runs.

`scripts/train.py` writes `outputs/results.csv` plus reusable artifacts in
`outputs/models/`. Run it before the results or demo notebooks if those files are absent.

The original contribution is the data-efficiency comparison using 10%, 25%, and 100%
of the selected training data.
