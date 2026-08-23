from pathlib import Path
import csv

import matplotlib.pyplot as plt


LOGS_DIR = Path(__file__).resolve().parent
CSV_FILES = sorted(LOGS_DIR.glob("metrics_*.csv"))
METRICS = ("train_loss", "val_loss", "lr")


def load_metrics(csv_path):
    data = {metric: [] for metric in ("step", *METRICS)}

    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            for metric in data:
                data[metric].append(float(row[metric]))

    return data


def main():
    if not CSV_FILES:
        raise FileNotFoundError(f"Aucun fichier metrics_*.csv trouve dans {LOGS_DIR}")

    for csv_path in CSV_FILES:
        data = load_metrics(csv_path)
        label_prefix = csv_path.stem.replace("metrics_", "")
        fig, ax_loss = plt.subplots(figsize=(12, 7))
        ax_lr = ax_loss.twinx()

        ax_loss.plot(
            data["step"],
            data["train_loss"],
            label="train_loss",
        )
        ax_loss.plot(
            data["step"],
            data["val_loss"],
            linestyle="--",
            label="val_loss",
        )
        ax_lr.plot(
            data["step"],
            data["lr"],
            linestyle=":",
            alpha=0.8,
            label="lr",
        )

        ax_loss.set_title(f"Metrics d'entrainement - {label_prefix}")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_lr.set_ylabel("Learning rate")
        ax_loss.grid(True, alpha=0.3)

        lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
        lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
        ax_loss.legend(
            lines_loss + lines_lr,
            labels_loss + labels_lr,
            loc="upper right",
            fontsize=8,
        )

        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
