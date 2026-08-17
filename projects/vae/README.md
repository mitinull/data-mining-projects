# MNIST β-VAE: Generation and Latent Space

This VAE-1 project compares matched 2-D latent MNIST VAEs at β = 0.5, 1, and 4. Its original contribution is a quantitative reconstruction-versus-regularization study: test reconstruction BCE, KL, ELBO, latent k-NN digit accuracy, and generated-image diversity are measured under the same architecture and training protocol.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/train.py
    python scripts/build_report_assets.py

The first run downloads MNIST and writes real artifacts under `outputs/`. Use `--epochs 1` only as a smoke test; the default β sweep is the full experiment. Notebooks 2 and 3 have an optional training cell—skip it when outputs already exist. Compile the report with `cd report && pdflatex -interaction=nonstopmode report.tex`.
