"""Train reproducible RNN forecasters on the Jena Climate dataset."""
import argparse
import csv
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
TARGETS = ["T (degC)", "rh (%)"]
HORIZONS = [6, 24, 72]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def download_data(data_dir):
    data_dir = Path(data_dir)
    csv_path = data_dir / "jena_climate_2009_2016.csv"
    if csv_path.exists():
        return csv_path
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "jena_climate_2009_2016.csv.zip"
    print(f"Downloading Jena Climate data to {archive} ...")
    urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        member = next(name for name in zipped.namelist() if name.endswith(".csv"))
        with zipped.open(member) as source, csv_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    archive.unlink(missing_ok=True)
    return csv_path


def load_hourly_data(data_dir):
    frame = pd.read_csv(download_data(data_dir))
    frame["Date Time"] = pd.to_datetime(frame["Date Time"], format="%d.%m.%Y %H:%M:%S")
    feature_columns = [column for column in frame.columns if column != "Date Time"]
    frame[feature_columns] = frame[feature_columns].replace(-9999.0, np.nan)
    # Original data is sampled every ten minutes; retain each full-hour observation.
    frame = frame.iloc[::6].reset_index(drop=True)
    return frame, feature_columns


def prepare_data(data_dir):
    frame, feature_columns = load_hourly_data(data_dir)
    n_rows = len(frame)
    train_end = int(n_rows * 0.70)
    validation_end = int(n_rows * 0.85)
    values = frame[feature_columns].astype("float32")
    medians = values.iloc[:train_end].median()
    values = values.fillna(medians)
    mean = values.iloc[:train_end].mean()
    std = values.iloc[:train_end].std().replace(0, 1)
    normalized = ((values - mean) / std).to_numpy(dtype=np.float32)
    target_indices = [feature_columns.index(target) for target in TARGETS]
    return {
        "values": normalized,
        "raw_targets": values[TARGETS].to_numpy(dtype=np.float32),
        "dates": frame["Date Time"].to_numpy(),
        "feature_columns": feature_columns,
        "target_indices": target_indices,
        "target_mean": mean[TARGETS].to_numpy(dtype=np.float32),
        "target_std": std[TARGETS].to_numpy(dtype=np.float32),
        "train_end": train_end,
        "validation_end": validation_end,
    }


class WindowDataset(Dataset):
    def __init__(self, values, start, end, window, horizon, target_indices, stride=1):
        self.values = values
        self.indices = np.arange(max(start, window), end - horizon, stride)
        self.window = window
        self.horizon = horizon
        self.target_indices = target_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        history = self.values[index - self.window:index]
        future = self.values[index:index + self.horizon, self.target_indices]
        return torch.from_numpy(history), torch.from_numpy(future)


class DirectForecaster(nn.Module):
    def __init__(self, cell, input_size, hidden_size, horizon, output_size):
        super().__init__()
        self.rnn = cell(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon * output_size)
        self.horizon = horizon
        self.output_size = output_size

    def forward(self, inputs):
        _, state = self.rnn(inputs)
        if isinstance(state, tuple):
            state = state[0]
        return self.head(state[-1]).reshape(-1, self.horizon, self.output_size)


class Seq2SeqForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, horizon, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)
        self.horizon = horizon
        self.output_size = output_size

    def forward(self, inputs):
        _, state = self.encoder(inputs)
        zeros = torch.zeros(inputs.size(0), self.horizon, self.output_size, device=inputs.device)
        decoded, _ = self.decoder(zeros, state)
        return self.head(decoded)


class AttentionSeq2SeqForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, horizon, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTMCell(hidden_size + output_size, hidden_size)
        self.head = nn.Linear(hidden_size * 2, output_size)
        self.horizon = horizon
        self.output_size = output_size

    def forward(self, inputs):
        encoded, (hidden, cell) = self.encoder(inputs)
        hidden, cell = hidden[-1], cell[-1]
        previous = torch.zeros(inputs.size(0), self.output_size, device=inputs.device)
        predictions = []
        for _ in range(self.horizon):
            weights = torch.softmax(torch.bmm(encoded, hidden.unsqueeze(2)).squeeze(2), dim=1)
            context = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)
            hidden, cell = self.decoder(torch.cat([previous, context], dim=1), (hidden, cell))
            previous = self.head(torch.cat([hidden, context], dim=1))
            predictions.append(previous)
        return torch.stack(predictions, dim=1)


def make_loaders(prepared, window, horizon, batch_size, stride):
    values = prepared["values"]
    train = WindowDataset(values, 0, prepared["train_end"], window, horizon, prepared["target_indices"], stride)
    validation = WindowDataset(values, prepared["train_end"], prepared["validation_end"], window, horizon, prepared["target_indices"], stride)
    test = WindowDataset(values, prepared["validation_end"], len(values), window, horizon, prepared["target_indices"], stride)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(validation, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        test.indices,
    )


def loss_and_predictions(model, loader, device):
    model.eval()
    all_predictions, all_targets, losses = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            losses.append(torch.mean((predictions - targets) ** 2).item())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(all_predictions), np.concatenate(all_targets)


def train_model(name, model, train_loader, validation_loader, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state, best_loss, stale_epochs, history = None, float("inf"), 0, []
    model.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = torch.mean((model(inputs) - targets) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        validation_loss, _, _ = loss_and_predictions(model, validation_loader, device)
        history.append({"model": name, "epoch": epoch, "train_mse": float(np.mean(train_losses)), "validation_mse": validation_loss})
        print(f"{name}: epoch {epoch:02d}, train MSE {history[-1]['train_mse']:.4f}, validation MSE {validation_loss:.4f}")
        if validation_loss < best_loss - 1e-5:
            best_loss, stale_epochs = validation_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    return model.cpu(), history


def denormalize(values, prepared):
    return values * prepared["target_std"] + prepared["target_mean"]


def metric_rows(name, prediction, target):
    rows = []
    for hours in HORIZONS:
        index = hours - 1
        for target_index, target_name in enumerate(TARGETS):
            errors = prediction[:, index, target_index] - target[:, index, target_index]
            rows.append({"model": name, "horizon_hours": hours, "target": target_name,
                         "mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2)))})
    return rows


def save_outputs(output_dir, rows, history, predictions, targets, test_indices, prepared, models, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "horizon_hours", "target", "mae", "rmse"])
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "epoch", "train_mse", "validation_mse"])
        writer.writeheader(); writer.writerows(history)
    np.savez_compressed(output_dir / "predictions.npz", targets=targets, indices=test_indices,
                        dates=prepared["dates"][test_indices], **predictions)
    model_dir = output_dir / "models"; model_dir.mkdir(exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), model_dir / f"{name}.pt")
    with (output_dir / "config.json").open("w") as file:
        json.dump(vars(args) | {"targets": TARGETS, "horizons_hours": HORIZONS}, file, indent=2, default=str)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--window-hours", type=int, default=72)
    parser.add_argument("--window-stride-hours", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=["lstm", "gru", "seq2seq_lstm", "attention_seq2seq"],
                        help="Train only selected models; repeated invocations merge their saved artifacts.")
    args = parser.parse_args()
    seed_everything(args.seed)
    prepared = prepare_data(args.data_dir)
    train_loader, validation_loader, test_loader, test_indices = make_loaders(prepared, args.window_hours, max(HORIZONS), args.batch_size, args.window_stride_hours)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    constructors = {
        "lstm": lambda: DirectForecaster(nn.LSTM, len(prepared["feature_columns"]), args.hidden_size, max(HORIZONS), len(TARGETS)),
        "gru": lambda: DirectForecaster(nn.GRU, len(prepared["feature_columns"]), args.hidden_size, max(HORIZONS), len(TARGETS)),
        "seq2seq_lstm": lambda: Seq2SeqForecaster(len(prepared["feature_columns"]), args.hidden_size, max(HORIZONS), len(TARGETS)),
        "attention_seq2seq": lambda: AttentionSeq2SeqForecaster(len(prepared["feature_columns"]), args.hidden_size, max(HORIZONS), len(TARGETS)),
    }
    selected = args.models or list(constructors)
    all_targets, prediction_sets, rows, history, trained = None, {}, [], [], {}
    if (args.output_dir / "predictions.npz").exists():
        previous = np.load(args.output_dir / "predictions.npz")
        all_targets = previous["targets"]
        for name in previous.files:
            if name not in {"targets", "indices", "dates", "persistence"}:
                prediction_sets[name] = previous[name]
        rows = pd.read_csv(args.output_dir / "results.csv").to_dict("records")
        history = pd.read_csv(args.output_dir / "history.csv").to_dict("records")
        rows = [row for row in rows if row["model"] not in set(selected) | {"persistence"}]
        history = [row for row in history if row["model"] not in set(selected)]
    for name in selected:
        constructor = constructors[name]
        model, model_history = train_model(name, constructor(), train_loader, validation_loader, args, device)
        _, predictions, targets = loss_and_predictions(model, test_loader, torch.device("cpu"))
        predictions, targets = denormalize(predictions, prepared), denormalize(targets, prepared)
        prediction_sets[name] = predictions; all_targets = targets
        rows.extend(metric_rows(name, predictions, targets)); history.extend(model_history); trained[name] = model
    persistence = np.repeat(all_targets[:, :1, :], max(HORIZONS), axis=1)
    # Replace persistence with the last observed target values for each test window.
    last_observed = denormalize(prepared["values"][test_indices - 1][:, prepared["target_indices"]], prepared)
    persistence = np.repeat(last_observed[:, None, :], max(HORIZONS), axis=1)
    prediction_sets["persistence"] = persistence
    rows.extend(metric_rows("persistence", persistence, all_targets))
    save_outputs(args.output_dir, rows, history, prediction_sets, all_targets, test_indices, prepared, trained, args)
    print(f"Saved {len(rows)} metric rows to {args.output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
