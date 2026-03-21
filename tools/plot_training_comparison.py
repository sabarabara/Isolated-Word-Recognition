#!/usr/bin/env python3
import glob
import os
import csv
import math
from collections import OrderedDict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_PNG = os.path.join(RESULTS_DIR, "training_comparison.png")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "training_summary.csv")

# candidate column names for validation top-1 accuracy
CANDIDATES = [
    "val_acc_top1",
    "val_acc_top_1",
    "val_top_1",
    "val_acc_1",
    "val_accuracy",
    "val_acc",
    "val_acc_top_1%",
    "val_1",
]


def find_training_logs():
    pattern = os.path.join(RESULTS_DIR, "*", "metrics", "training_log.csv")
    return sorted(glob.glob(pattern))


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames if "reader" in locals() else None, rows


def pick_val_column(fieldnames):
    if not fieldnames:
        return None
    lower = [f.lower() for f in fieldnames]
    for cand in CANDIDATES:
        if cand in lower:
            # return the actual casing from fieldnames
            return fieldnames[lower.index(cand)]
    # fallback heuristics: any field that contains 'val' and ('acc' or 'top')
    for f in fieldnames:
        fn = f.lower()
        if "val" in fn and ("acc" in fn or "top" in fn or "accuracy" in fn):
            return f
    # if nothing, try 'accuracy'
    for f in fieldnames:
        if "accuracy" in f.lower():
            return f
    return None


def to_float(x):
    try:
        return float(x)
    except Exception:
        # try strip percent
        if isinstance(x, str) and x.strip().endswith("%"):
            try:
                return float(x.strip().rstrip("%"))
            except Exception:
                return math.nan
        return math.nan


def main():
    logs = find_training_logs()
    if not logs:
        print("No training_log.csv found under results/*/metrics/")
        return 1

    series = OrderedDict()
    summary = []

    for p in logs:
        # model name is parent of metrics folder
        model_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        fieldnames, rows = read_csv(p)
        if not rows:
            print(f"skipping empty {p}")
            continue
        val_col = pick_val_column(fieldnames)
        epoch_col = None
        for cand in ["epoch", "Epoch", "ep"]:
            if cand in (fieldnames or []):
                epoch_col = cand
                break
        if epoch_col is None:
            epoch_col = fieldnames[0]

        xs = []
        ys = []
        for r in rows:
            x = to_float(r.get(epoch_col, ""))
            y = to_float(r.get(val_col, "")) if val_col else math.nan
            if math.isnan(x):
                # try index-based epoch
                try:
                    x = len(xs) + 1
                except Exception:
                    x = None
            xs.append(x)
            ys.append(y)

        # store
        series[model_name] = (xs, ys, val_col)
        # final metric
        final_idx = None
        for i in range(len(ys) - 1, -1, -1):
            if not math.isnan(ys[i]):
                final_idx = i
                break
        final_val = ys[final_idx] if final_idx is not None else math.nan
        summary.append(
            {"model": model_name, "final_val": final_val, "val_column": val_col or ""}
        )

    # plotting
    plt.figure(figsize=(10, 6))
    for model, (xs, ys, col) in series.items():
        # convert to numeric lists
        xs_num = [float(x) if x is not None else i + 1 for i, x in enumerate(xs)]
        ys_num = [
            float(y) if (y is not None and not math.isnan(y)) else math.nan for y in ys
        ]
        plt.plot(xs_num, ys_num, marker="o", label=f"{model} ({col})")

    plt.xlabel("epoch")
    plt.ylabel("validation top-1 (percent)")
    plt.title("Validation Top-1 comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(OUT_PNG)
    print(f"Saved plot to {OUT_PNG}")

    # write summary CSV
    with open(OUT_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "val_column", "final_val"])
        writer.writeheader()
        for s in summary:
            writer.writerow(
                {
                    "model": s["model"],
                    "val_column": s["val_column"],
                    "final_val": s["final_val"],
                }
            )
    print(f"Saved summary to {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
