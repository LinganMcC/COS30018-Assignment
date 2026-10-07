import csv
from pathlib import Path
from datetime import datetime


# Standard location, relative to the repo root. Created on first write.
LOG_PATH = Path("experiments") / "experiment_log.csv"

# Column order (Need to be dicussed before changed)
FIELDS = [
    "timestamp",
    "run_id",
    "architecture",
    "learning_rate",
    "batch_size",
    "epochs",
    "dropout",
    "val_accuracy",
    "test_accuracy",
    "training_time_s",
    "notes",
]


def log_run(
    run_id: str,
    architecture: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    dropout: float,
    test_accuracy: float,
    training_time_s: float,
    val_accuracy: float | None = None,
    notes: str = "",
    log_path: Path | str | None = None,
) -> None:
    """Append one row to the experiment log. Creates the file + header if needed."""
    # Look up LOG_PATH at call time (not as a default arg) so tests/callers can
    # override it by reassigning experiment_logger.LOG_PATH.
    if log_path is None:
        log_path = LOG_PATH
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    new_file = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "architecture": architecture,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "dropout": dropout,
            "val_accuracy": f"{val_accuracy:.4f}" if val_accuracy is not None else "",
            "test_accuracy": f"{test_accuracy:.4f}",
            "training_time_s": f"{training_time_s:.2f}",
            "notes": notes,
        })
    print(f"Logged run '{run_id}' to {log_path}")
