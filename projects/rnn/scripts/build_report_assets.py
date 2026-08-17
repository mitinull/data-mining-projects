"""Create report figures and LaTeX fragments from real saved experiment artifacts."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from train import HORIZONS, TARGETS, load_hourly_data


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"


def save_eda_figures():
    frame, features = load_hourly_data(ROOT / "data")
    FIGURES.mkdir(parents=True, exist_ok=True)
    sample = frame.iloc[:24 * 31]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True)
    axes[0].plot(sample["Date Time"], sample["T (degC)"], color="#4C78A8", linewidth=.8)
    axes[0].set_ylabel("Temperature (°C)")
    axes[1].plot(sample["Date Time"], sample["rh (%)"], color="#E45756", linewidth=.8)
    axes[1].set_ylabel("Relative humidity (%)")
    axes[1].set_xlabel("Date (first month in record)")
    fig.suptitle("Hourly Jena Climate target signals")
    fig.tight_layout(); fig.savefig(FIGURES / "target_timeseries.png", dpi=180); plt.close(fig)

    correlations = frame[features].replace(-9999, np.nan).corr()[TARGETS].sort_values("T (degC)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    for axis, target, color in zip(axes, TARGETS, ["#4C78A8", "#E45756"]):
        correlations[target].plot.barh(ax=axis, color=color)
        axis.set_title(f"Correlation with {target}"); axis.set_xlabel("Pearson r")
    fig.tight_layout(); fig.savefig(FIGURES / "correlations.png", dpi=180); plt.close(fig)


def save_result_figures(results, predictions):
    order = ["persistence", "lstm", "gru", "seq2seq_lstm", "attention_seq2seq"]
    labels = ["Persistence", "LSTM", "GRU", "Seq2Seq", "Attention Seq2Seq"]
    palette = ["#9D755D", "#4C78A8", "#59A14F", "#F28E2B", "#E45756"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for axis, target in zip(axes, TARGETS):
        for model, label, color in zip(order, labels, palette):
            rows = results[(results.model == model) & (results.target == target)].sort_values("horizon_hours")
            axis.plot(rows.horizon_hours, rows.mae, marker="o", label=label, color=color)
        axis.set_title(f"{target}: MAE by horizon"); axis.set_xlabel("Forecast horizon (hours)"); axis.set_ylabel("MAE")
        axis.set_xticks(HORIZONS); axis.grid(alpha=.3)
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(FIGURES / "mae_by_horizon.png", dpi=180); plt.close(fig)

    history = pd.read_csv(OUTPUTS / "history.csv")
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    for model, group in history.groupby("model"):
        axis.plot(group.epoch, group.validation_mse, marker="o", label=model.replace("_", " "))
    axis.set(xlabel="Epoch", ylabel="Validation MSE", title="Validation learning curves")
    axis.legend(fontsize=8); axis.grid(alpha=.3); fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png", dpi=180); plt.close(fig)

    targets = predictions["targets"]
    dates = pd.to_datetime(predictions["dates"])
    chosen = np.linspace(0, len(targets) - 1, 3, dtype=int)
    models = ["lstm", "gru", "seq2seq_lstm", "attention_seq2seq", "persistence"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.8), sharex=True)
    for column, index in enumerate(chosen):
        hours = np.arange(1, 73)
        for row, (target_index, target_name) in enumerate(enumerate(TARGETS)):
            axis = axes[row, column]
            axis.plot(hours, targets[index, :, target_index], color="black", linewidth=2, label="Observed")
            for model, color in zip(models, ["#4C78A8", "#59A14F", "#F28E2B", "#E45756", "#9D755D"]):
                axis.plot(hours, predictions[model][index, :, target_index], linewidth=1, color=color, label=model)
            axis.set_title(str(dates[index])[:10]); axis.set_ylabel(target_name); axis.grid(alpha=.25)
            if row == 1: axis.set_xlabel("Hours ahead")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(FIGURES / "forecast_examples.png", dpi=180); plt.close(fig)


def write_tex(results):
    labels = {"persistence": "Persistence", "lstm": "LSTM", "gru": "GRU", "seq2seq_lstm": "Seq2Seq LSTM", "attention_seq2seq": "Attention Seq2Seq"}
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Horizon & Model & T MAE & T RMSE & RH MAE & RH RMSE \\", r"\midrule"]
    for horizon in HORIZONS:
        for model in labels:
            subset = results[(results.model == model) & (results.horizon_hours == horizon)].set_index("target")
            temperature, humidity = subset.loc[TARGETS[0]], subset.loc[TARGETS[1]]
            lines.append(f"{horizon} h & {labels[model]} & {temperature.mae:.3f} & {temperature.rmse:.3f} & {humidity.mae:.3f} & {humidity.rmse:.3f} " + r"\\")
        if horizon != HORIZONS[-1]: lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (REPORT / "generated_results.tex").write_text("\n".join(lines) + "\n")

    rnn = results[results.model != "persistence"]
    short = rnn[rnn.horizon_hours == 6].groupby("model").mae.mean().idxmin()
    long = rnn[rnn.horizon_hours == 72].groupby("model").mae.mean().idxmin()
    seq = rnn[rnn.model == "seq2seq_lstm"].groupby("horizon_hours").mae.mean()
    attention = rnn[rnn.model == "attention_seq2seq"].groupby("horizon_hours").mae.mean()
    gain_short = 100 * (seq.loc[6] - attention.loc[6]) / seq.loc[6]
    gain_long = 100 * (seq.loc[72] - attention.loc[72]) / seq.loc[72]
    summary = (
        f"Across the two targets, {labels[short]} has the lowest mean RNN MAE at 6 hours and "
        f"{labels[long]} has the lowest mean RNN MAE at 72 hours. Relative to the matched "
        f"Seq2Seq LSTM, attention changes mean MAE by {gain_short:+.1f}\\% at 6 hours and "
        f"{gain_long:+.1f}\\% at 72 hours."
    )
    (REPORT / "generated_summary.tex").write_text(summary + "\n")


def main():
    if not (OUTPUTS / "results.csv").exists():
        raise FileNotFoundError("Run python scripts/train.py before building report assets.")
    results = pd.read_csv(OUTPUTS / "results.csv")
    predictions = np.load(OUTPUTS / "predictions.npz")
    save_eda_figures(); save_result_figures(results, predictions); write_tex(results)
    print(f"Report assets written to {REPORT}")


if __name__ == "__main__":
    main()
