"""Train normal-only Conv1D and LSTM autoencoders on ECG5000."""

import argparse
import csv
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA_URLS = (
    "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip",
    "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ECG5000.zip",
)
RAW_URLS = {
    "ECG5000_TRAIN.txt": "https://raw.githubusercontent.com/kanesp/ECG_Anomaly-Detection/main/data/ECG5000_TRAIN.txt",
    "ECG5000_TEST.txt": "https://raw.githubusercontent.com/kanesp/ECG_Anomaly-Detection/main/data/ECG5000_TEST.txt",
}
SEQUENCE_LENGTH = 140
NORMAL_LABEL = 1


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.to_latent = nn.Linear(32 * 35, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 32 * 35)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, inputs):
        latent = self.to_latent(self.encoder(inputs).flatten(1))
        return self.decoder(self.from_latent(latent).view(-1, 32, 35))


class LSTMAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_size=32):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_size, batch_first=True)
        self.to_signal = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        _, (hidden, _) = self.encoder(inputs.transpose(1, 2))
        latent = self.to_latent(hidden[-1])
        repeated = latent.unsqueeze(1).expand(-1, SEQUENCE_LENGTH, -1)
        decoded, _ = self.decoder(repeated)
        return self.to_signal(decoded).transpose(1, 2)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_data(data_dir):
    data_dir = Path(data_dir)
    train_file, test_file = data_dir / "ECG5000_TRAIN.txt", data_dir / "ECG5000_TEST.txt"
    if train_file.exists() and test_file.exists():
        return train_file, test_file
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        print("Downloading ECG5000 text files from the public mirror")
        for destination in (train_file, test_file):
            urllib.request.urlretrieve(RAW_URLS[destination.name], destination)
        return train_file, test_file
    except Exception:
        train_file.unlink(missing_ok=True)
        test_file.unlink(missing_ok=True)
    archive_path = data_dir / "ECG5000.zip"
    error = None
    for url in DATA_URLS:
        try:
            print(f"Downloading ECG5000 from {url}")
            urllib.request.urlretrieve(url, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                for destination in (train_file, test_file):
                    member = next(name for name in archive.namelist() if name.endswith(destination.name))
                    with archive.open(member) as source, destination.open("wb") as target:
                        target.write(source.read())
            return train_file, test_file
        except Exception as caught:
            error = caught
            archive_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not download ECG5000. Put ECG5000_TRAIN.txt and ECG5000_TEST.txt in {data_dir}."
    ) from error


def load_data(data_dir):
    train_file, test_file = download_data(data_dir)
    train, test = np.loadtxt(train_file), np.loadtxt(test_file)
    x_train, y_train = train[:, 1:].astype("float32"), train[:, 0].astype(int)
    x_test, y_test = test[:, 1:].astype("float32"), test[:, 0].astype(int)
    if x_train.shape[1] != SEQUENCE_LENGTH or x_test.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("ECG5000 sequences must have length 140.")
    return x_train, y_train, x_test, y_test


def prepare_data(x_train, y_train, x_test, y_test, seed):
    normals = x_train[y_train == NORMAL_LABEL]
    train_normal, validation_normal = train_test_split(normals, test_size=0.2, random_state=seed)
    mean, std = train_normal.mean(0), train_normal.std(0)
    std[std < 1e-6] = 1.0
    scale = lambda x: ((x - mean) / std).astype("float32")
    return {
        "train_normal": scale(train_normal), "validation_normal": scale(validation_normal),
        "test": scale(x_test), "test_anomaly": (y_test != NORMAL_LABEL).astype(int),
        "test_raw_label": y_test, "mean": mean, "std": std,
    }


def loader(values, batch_size, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(values).unsqueeze(1)), batch_size=batch_size, shuffle=shuffle)


def errors_and_reconstructions(model, values, batch_size):
    errors, reconstructions = [], []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader(values, batch_size):
            output = model(batch)
            errors.extend(((output - batch) ** 2).mean((1, 2)).tolist())
            reconstructions.append(output.squeeze(1).numpy())
    return np.asarray(errors), np.concatenate(reconstructions)


def train_model(model, train_values, validation_values, args):
    optimizer, loss_fn = torch.optim.AdamW(model.parameters(), lr=args.learning_rate), nn.MSELoss()
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (batch,) in loader(train_values, args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        validation_errors, _ = errors_and_reconstructions(model, validation_values, args.batch_size)
        validation_loss = float(validation_errors.mean())
        print(f"{model.__class__.__name__} epoch {epoch:02d}: train={np.mean(losses):.5f}, validation={validation_loss:.5f}")
        if validation_loss < best_loss - 1e-6:
            best_loss, stale = validation_loss, 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)


def evaluate(name, model, prepared, batch_size):
    validation_errors, _ = errors_and_reconstructions(model, prepared["validation_normal"], batch_size)
    median = float(np.median(validation_errors))
    mad = float(np.median(np.abs(validation_errors - median)))
    threshold = median + 3 * 1.4826 * mad
    test_errors, reconstructions = errors_and_reconstructions(model, prepared["test"], batch_size)
    labels, predictions = prepared["test_anomaly"], (test_errors > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": name, "threshold": threshold, "validation_median_error": median, "validation_mad": mad,
        "precision": precision, "recall": recall, "f1": f1,
        "average_precision": average_precision_score(labels, test_errors), "roc_auc": roc_auc_score(labels, test_errors),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "errors": test_errors, "predictions": predictions, "reconstructions": reconstructions,
    }


def save_outputs(results, prepared, models, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    fields = ["model", "threshold", "validation_median_error", "validation_mad", "precision", "recall", "f1", "average_precision", "roc_auc", "tn", "fp", "fn", "tp"]
    with (output_dir / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: result[field] for field in fields} for result in results])
    rows = [
        {"model": result["model"], "test_index": index, "raw_label": int(prepared["test_raw_label"][index]),
         "is_anomaly": int(prepared["test_anomaly"][index]), "reconstruction_error": float(error),
         "prediction": int(prediction)}
        for result in results for index, (error, prediction) in enumerate(zip(result["errors"], result["predictions"]))
    ]
    with (output_dir / "test_errors.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    arrays = {"test": prepared["test"], "test_anomaly": prepared["test_anomaly"], "test_raw_label": prepared["test_raw_label"], "mean": prepared["mean"], "std": prepared["std"]}
    for result in results:
        arrays[f"{result['model']}_reconstruction"] = result["reconstructions"]
        arrays[f"{result['model']}_error"] = result["errors"]
    np.savez(output_dir / "demo_data.npz", **arrays)
    for name, model in models.items():
        torch.save(model.state_dict(), output_dir / "models" / f"{name}.pt")


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    x_train, y_train, x_test, y_test = load_data(args.data_dir)
    prepared = prepare_data(x_train, y_train, x_test, y_test, args.seed)
    print(f"Normal training beats: {len(prepared['train_normal'])}; normal validation beats: {len(prepared['validation_normal'])}; test beats: {len(prepared['test'])}.")
    models = {"conv_ae": ConvAutoencoder(), "lstm_ae": LSTMAutoencoder()}
    results = []
    for name, model in models.items():
        train_model(model, prepared["train_normal"], prepared["validation_normal"], args)
        result = evaluate(name, model, prepared, args.batch_size)
        results.append(result)
        print(f"{name}: precision={result['precision']:.3f}, recall={result['recall']:.3f}, f1={result['f1']:.3f}, threshold={result['threshold']:.5f}")
    save_outputs(results, prepared, models, args.output_dir)


if __name__ == "__main__":
    main()
