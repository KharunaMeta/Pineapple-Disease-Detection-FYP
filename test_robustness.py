"""
Batch robustness evaluation across environmental-condition folders (Sprint-style metrics).

Expected layout (example):
  data_root/
    luminance/
      extreme/
        Healthy/*.jpg
        Heart_Rot/*.png
      low/
        Healthy/
        Heart_Rot/
    blur/
      mild/
        ...

Class folder names (case-insensitive): Healthy, Heart_Rot (aliases: heart_rot, HeartRot).
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from model_builder import CLASS_NAMES, infer, load_or_build_model
from preprocessing import preprocess_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Canonical label -> class index
_CLASS_ALIASES = {
    "healthy": 0,
    "heart_rot": 1,
    "heartrot": 1,
    "heart-rot": 1,
    "disease": 1,
}


def _class_index_from_folder(name: str) -> int | None:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in _CLASS_ALIASES:
        return _CLASS_ALIASES[key]
    if key == CLASS_NAMES[0].lower():
        return 0
    if key == CLASS_NAMES[1].lower():
        return 1
    return None


def _iter_image_files(class_dir: Path):
    for p in sorted(class_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def collect_pairs(condition_root: Path) -> list[tuple[Path, int]]:
    """condition_root contains subfolders per class with images."""
    pairs: list[tuple[Path, int]] = []
    if not condition_root.is_dir():
        return pairs
    for child in sorted(condition_root.iterdir()):
        if not child.is_dir():
            continue
        cid = _class_index_from_folder(child.name)
        if cid is None:
            continue
        for img_path in _iter_image_files(child):
            pairs.append((img_path, cid))
    return pairs


def run_condition(
    model,
    paths: list[Path],
    y_true: list[int],
    *,
    use_histogram_equalization: bool,
    normalize_mode: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns y_true_arr, y_pred_arr, n_skipped (unreadable)."""
    y_true_list: list[int] = []
    pred_list: list[int] = []
    skipped = 0

    batch_x: list[np.ndarray] = []
    batch_y: list[int] = []

    for path, yt in zip(paths, y_true):
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                skipped += 1
                continue
            x = preprocess_image(
                bgr,
                use_histogram_equalization=use_histogram_equalization,
                normalize_mode=normalize_mode,
            )
            batch_x.append(x[0])
            batch_y.append(yt)
            if len(batch_x) >= batch_size:
                probs = infer(model, np.stack(batch_x, axis=0))
                for row, yv in zip(probs, batch_y):
                    pred_list.append(int(np.argmax(row)))
                    y_true_list.append(yv)
                batch_x = []
                batch_y = []
        except Exception:
            skipped += 1

    if batch_x:
        probs = infer(model, np.stack(batch_x, axis=0))
        for row, yv in zip(probs, batch_y):
            pred_list.append(int(np.argmax(row)))
            y_true_list.append(yv)

    if not y_true_list:
        return np.array([]), np.array([]), skipped
    return np.array(y_true_list), np.array(pred_list), skipped


def print_metrics(variable: str, intensity: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    prec = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n=== {variable} / {intensity} ===")
    print(f"Samples: {len(y_true)}")
    print(f"Precision (Heart_Rot positive): {prec:.4f}")
    print(f"Recall    (Heart_Rot positive): {rec:.4f}")
    print(f"F1-score  (Heart_Rot positive): {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix [rows=true Healthy,Heart_Rot | cols=pred]:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    return {
        "variable": variable,
        "intensity": intensity,
        "n_samples": len(y_true),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def discover_runs(data_root: Path) -> list[tuple[str, str, list[tuple[Path, int]]]]:
    """
    Each run = (variable, intensity, list of (path, y_true)).
    data_root/<variable>/<intensity>/<ClassName>/file
    """
    runs: list[tuple[str, str, list[tuple[Path, int]]]] = []
    if not data_root.is_dir():
        return runs
    for var_dir in sorted(data_root.iterdir()):
        if not var_dir.is_dir():
            continue
        variable = var_dir.name
        for int_dir in sorted(var_dir.iterdir()):
            if not int_dir.is_dir():
                continue
            intensity = int_dir.name
            pairs = collect_pairs(int_dir)
            if pairs:
                runs.append((variable, intensity, pairs))
    return runs


def main():
    parser = argparse.ArgumentParser(description="Robustness batch testing for MobileNetV2 binary classifier.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder with variable/intensity/class layout.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_WEIGHTS_PATH", ""),
        help="Path to .keras/.h5 weights or full model (optional; uses env MODEL_WEIGHTS_PATH if unset).",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("robustness_results.csv"))
    parser.add_argument("--histogram-eq", action="store_true", help="Enable luminance histogram equalization during tests.")
    parser.add_argument("--normalize", choices=("mobilenet", "div255"), default="mobilenet")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model, loaded = load_or_build_model(args.model or None)
    if not loaded:
        print("Warning: running with freshly initialized head — metrics are not meaningful until you train or supply weights.")

    runs = discover_runs(args.data_root)
    if not runs:
        print(f"No runs found under {args.data_root}. Expected: <variable>/<intensity>/Healthy/ and .../Heart_Rot/")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "variable",
        "intensity",
        "histogram_equalization",
        "normalize_mode",
        "n_samples",
        "n_skipped_unreadable",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
        "model_weights_loaded",
    ]
    write_header = not args.output_csv.is_file()
    ts = datetime.now(timezone.utc).isoformat()

    with args.output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for variable, intensity, pairs in runs:
            paths = [p for p, _ in pairs]
            y_true_arr = np.array([y for _, y in pairs], dtype=np.int32)
            y_true, y_pred, skipped = run_condition(
                model,
                paths,
                y_true_arr.tolist(),
                use_histogram_equalization=args.histogram_eq,
                normalize_mode=args.normalize,
                batch_size=args.batch_size,
            )
            if len(y_true) == 0:
                print(f"\n=== {variable} / {intensity} ===\nNo readable images (skipped {skipped}).")
                row = {
                    "timestamp_utc": ts,
                    "variable": variable,
                    "intensity": intensity,
                    "histogram_equalization": args.histogram_eq,
                    "normalize_mode": args.normalize,
                    "n_samples": 0,
                    "n_skipped_unreadable": skipped,
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "accuracy": "",
                    "tn": "",
                    "fp": "",
                    "fn": "",
                    "tp": "",
                    "model_weights_loaded": loaded,
                }
                writer.writerow(row)
                continue

            stats = print_metrics(variable, intensity, y_true, y_pred)
            stats.update(
                {
                    "timestamp_utc": ts,
                    "histogram_equalization": args.histogram_eq,
                    "normalize_mode": args.normalize,
                    "n_skipped_unreadable": skipped,
                    "model_weights_loaded": loaded,
                }
            )
            writer.writerow(stats)

    print(f"\nAppended results to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
