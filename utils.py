# -*- coding: utf-8 -*-
"""
utils.py  ( 5 seeds x 4 data sizes)
==============================================
Shared utilities for the IndoBERT PEFT comparison experiments
(FFT, LoRA, BitFit, IA3, Houlsby, Pfeiffer).

Import this SAME module in every method notebook:

    import sys
    sys.path.append('/content/drive/MyDrive/code')
    from utils import *

Experiment grid
---------------
6 methods x 4 DATA_SIZES (5 / 10 / 50 / 100 % of the training set) x 5 SEEDS.
Validation and test sets are ALWAYS the full, fixed ones (taken from the 100%
folder), so every cell is evaluated on identical data. Only train.csv changes
between data sizes.

Training budget (important design decision)
-------------------------------------------
The budget is defined in optimizer STEPS, not epochs:
    block_steps = steps in one epoch of the FULL training set (268 for 4,284 samples @ bs16)
    max_steps   = MAX_EPOCHS x block_steps               (4,020)
    evaluation / checkpoint / early-stopping check every `block_steps` steps
    patience    = EARLY_STOP_PATIENCE evaluations (= 3 x 268 steps)
At 100% data this is identical to the previous epoch-based protocol (one
evaluation per epoch, 15 epochs max, patience 3). At 5%/10%/50% it gives every
data size the same optimizer budget, so a small subset is not penalised by
receiving ~10x fewer gradient updates. In logs, "epochs_run" / "avg_epoch_s"
therefore mean "evaluation rounds" / "seconds per 268-step block" (true epochs
only at 100%).
"""

import os
import re
import gc
import csv
import json
import math
import time
import shutil
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# ═══════════════════════════════════════════════════════════════════════════
# FIXED CONFIGURATION -- identical across all six methods.
# Only learning_rate and the model_init function differ between methods.
# ═══════════════════════════════════════════════════════════════════════════

MODEL_NAME = "indobenchmark/indobert-base-p2"
MAX_LENGTH = 128
BATCH_SIZE = 16
MAX_EPOCHS = 15
EARLY_STOP_PATIENCE = 3   
LABEL2ID = {"fact": 0, "hoax": 1}
ID2LABEL = {0: "fact", 1: "hoax"}
SEEDS = [42, 123, 456, 789, 2026]
DATA_SIZES = [100, 50, 10, 5]
DEFAULT_TAG = "main"

# ── Paths -- adjust to your Drive layout ────────────────────────────────────
# Expected layout:
#   DATA_ROOT/            train.csv val.csv test.csv        (full data = 100%)
#   DATA_ROOT/50/         train.csv                          (50% subset)
#   DATA_ROOT/10/         train.csv                          (10% subset)
#   DATA_ROOT/5/         train.csv                          (5% subset)
DATA_ROOT = "/content/drive/MyDrive/data"
RESULTS_PATH = "/content/drive/MyDrive/results.csv"
PREDICTIONS_PATH = "/content/drive/MyDrive/predictions.csv"
LOGS_DIR = "/content/drive/MyDrive/logs"
RUNS_DIR = "/content/runs"            # local scratch (checkpoints deleted after each run)

RESULT_COLUMNS = [
    "method", "tag", "data_size", "seed", "train_samples",
    "accuracy", "precision", "recall", "f1",
    "trainable_params", "total_params", "trainable_pct", "learning_rate",
    "peak_vram_gb", "avg_epoch_s", "total_time_s",
    "epochs_run", "steps_run", "best_step", "best_val_f1",
    "gpu", "timestamp",
]
PRED_COLUMNS = ["method", "tag", "data_size", "seed", "index", "true_label", "predicted_label"]


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOCATION + LOADING
# ═══════════════════════════════════════════════════════════════════════════

def resolve_data_dir(size):
    size = int(size)
    candidates = [f"{DATA_ROOT}/{size}", f"{DATA_ROOT}/{size}pct",
                  f"{DATA_ROOT}/{size}percent", f"{DATA_ROOT}/{size}%"]
    if size == 100:
        candidates.append(DATA_ROOT)
    for d in candidates:
        if os.path.exists(f"{d}/train.csv"):
            return d
    raise FileNotFoundError(
        f"No train.csv found for data size {size}%. Looked in: " + ", ".join(candidates))


def _read_split(path):
    df = pd.read_csv(path)
    missing = {"berita", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; has {list(df.columns)}")
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].map(LABEL2ID)
    if df["label"].isna().any():
        raise ValueError(f"{path}: some labels could not be mapped to {LABEL2ID}")
    df["label"] = df["label"].astype(int)
    return df[["berita", "label"]].reset_index(drop=True)


def full_steps_per_epoch():
    """Steps in one epoch of the FULL (100%) training set -- the evaluation interval."""
    n = len(_read_split(f"{resolve_data_dir(100)}/train.csv"))
    return math.ceil(n / BATCH_SIZE)


def load_datasets(tokenizer, data_size=100):
    """Train subset from the size folder; val/test ALWAYS from the full folder."""
    full_dir = resolve_data_dir(100)
    train_df = _read_split(f"{resolve_data_dir(data_size)}/train.csv")
    val_df = _read_split(f"{full_dir}/val.csv")
    test_df = _read_split(f"{full_dir}/test.csv")

    def tokenize_fn(batch):
        return tokenizer(batch["berita"], truncation=True,
                         max_length=MAX_LENGTH)

    out = []
    for df in (train_df, val_df, test_df):
        ds = Dataset.from_pandas(df)
        ds = ds.map(tokenize_fn, batched=True)
        ds = ds.rename_column("label", "labels")
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        out.append(ds)
    return tuple(out)


def validate_data_sizes(sizes=None):
    sizes = sorted(sizes or DATA_SIZES)
    full_dir = resolve_data_dir(100)
    full_train = _read_split(f"{full_dir}/train.csv")
    val = _read_split(f"{full_dir}/val.csv")
    test = _read_split(f"{full_dir}/test.csv")
    full_beritas, val_beritas, test_beritas = set(full_train["berita"]), set(val["berita"]), set(test["berita"])

    rows, problems, warns = [], [], []
    prev_beritas, prev_size = None, None
    for s in sizes:
        d = resolve_data_dir(s)
        tr = _read_split(f"{d}/train.csv")
        beritas = set(tr["berita"])
        n_not_in_full = len(beritas - full_beritas)
        ov_val, ov_test = len(beritas & val_beritas), len(beritas & test_beritas)
        if n_not_in_full:
            problems.append(f"{s}%: {n_not_in_full} train beritas are not in the full train set")
        if ov_val or ov_test:
            problems.append(f"{s}%: train overlaps val ({ov_val}) / test ({ov_test}) -- leakage")
        if d != full_dir:
            for name, ref in (("val", val_beritas), ("test", test_beritas)):
                p = f"{d}/{name}.csv"
                if os.path.exists(p) and set(_read_split(p)["berita"]) != ref:
                    warns.append(f"{p} differs from the full {name}.csv; it is IGNORED "
                                 f"(the full one is always used)")
        if prev_beritas is not None and not prev_beritas <= beritas:
            warns.append(f"{prev_size}% subset is not nested inside the {s}% subset "
                         f"(works, but nested subsets make size comparisons cleaner)")
        rows.append({"size_%": s, "folder": d, "rows": len(tr),
                     "actual_%_of_full": round(100 * len(tr) / len(full_train), 1),
                     "hoax_%": round(100 * tr["label"].mean(), 1)})
        prev_beritas, prev_size = beritas, s

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nval rows: {len(val)} | test rows: {len(test)} | full train rows: {len(full_train)}")
    print(f"evaluation interval (steps per full-data epoch): {full_steps_per_epoch()}")
    for w in warns:
        print("WARNING:", w)
    if problems:
        raise AssertionError("Data problems:\n  - " + "\n  - ".join(problems))
    print("Data checks passed.")


def check_train_test_leakage():
    """Exact-duplicate beritas shared between the full train and test sets."""
    full_dir = resolve_data_dir(100)
    tr = _read_split(f"{full_dir}/train.csv")
    te = _read_split(f"{full_dir}/test.csv")
    overlap = set(tr["berita"]) & set(te["berita"])
    print(f"Exact duplicate beritas between train/test: {len(overlap)}")
    for t in list(overlap)[:5]:
        print(" -", t)
    return overlap


# ═══════════════════════════════════════════════════════════════════════════
# METRICS + TIMING
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class TimingCallback(TrainerCallback):
    """Pure training time per evaluation block (evaluation itself excluded),
    comparable with the old per-epoch timing."""

    def __init__(self, block_steps):
        self.block_steps = block_steps
        self.block_times = []
        self._t = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._t is not None and state.global_step % self.block_steps == 0:
            self.block_times.append(time.time() - self._t)

    def on_evaluate(self, args, state, control, **kwargs):
        self._t = time.time()


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING -- append-only, fixed schema. Never batch results in memory.
# ═══════════════════════════════════════════════════════════════════════════

def get_gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def run_key(method, data_size, seed, tag=DEFAULT_TAG):
    return f"{method}_{tag}_s{int(data_size)}_{int(seed)}"


def _check_header(path, expected):
    with open(path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != expected:
        raise RuntimeError(
            f"{path} uses an older column layout.\n"
            f"  found   : {header}\n  expected: {expected}\n"
            f"Run 00_Setup_and_Migration.ipynb (migrate_legacy_files) once before training.")


def _append_row(path, columns, row):
    extra = set(row) - set(columns)
    if extra:
        raise KeyError(f"Unknown columns in row: {extra}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    if exists:
        _check_header(path, columns)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            w.writeheader()
        w.writerow(row)


def log_result(row, results_path=None):
    _append_row(results_path or RESULTS_PATH, RESULT_COLUMNS, row)


def log_predictions(method, tag, data_size, seed, true_labels, preds, predictions_path=None):
    path = predictions_path or PREDICTIONS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    if exists:
        _check_header(path, PRED_COLUMNS)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(PRED_COLUMNS)
        for i, (t, p) in enumerate(zip(true_labels, preds)):
            w.writerow([method, tag, int(data_size), int(seed), i, int(t), int(p)])


def load_results(results_path=None):
    path = results_path or RESULTS_PATH
    _check_header(path, RESULT_COLUMNS)
    return pd.read_csv(path)


def already_done(method, data_size, seed, tag=DEFAULT_TAG, results_path=None):
    path = results_path or RESULTS_PATH
    if not os.path.exists(path):
        return False
    df = load_results(path)
    if df.empty:
        return False
    m = ((df["method"] == method) & (df["data_size"] == int(data_size)) &
         (df["seed"] == int(seed)) & (df["tag"] == tag))
    return bool(m.any())


def print_status(results_path=None, tag=DEFAULT_TAG):
    path = results_path or RESULTS_PATH
    if not os.path.exists(path):
        print("No results logged yet.")
        return
    df = load_results(path)
    df = df[(df["tag"] == tag) & (df["seed"].isin(SEEDS))]
    grid = (df.groupby(["method", "data_size"])["seed"].nunique().unstack(fill_value=0)
            .reindex(columns=sorted(DATA_SIZES), fill_value=0))
    print(f"Completed seeds per (method, data size); target = {len(SEEDS)}; tag='{tag}'")
    print(grid.to_string())
    print(f"Total: {int(grid.values.sum())} / {len(grid) * len(DATA_SIZES) * len(SEEDS) if len(grid) else 0}"
          f" for methods present")


# ═══════════════════════════════════════════════════════════════════════════
# CORE TRAINING LOOP -- identical for every method
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(method_name, seed, model_init_fn, learning_rate,
                   data_size=100, tag=DEFAULT_TAG,
                   output_dir_base=None, results_path=None, predictions_path=None,
                   skip_if_done=True):
    """
    model_init_fn: zero-arg callable returning a FRESH model. Called AFTER
                   set_seed(), so the classifier head's init is seeded too.
    data_size    : 5 / 10 / 50 / 100 (% of the training set; folder resolved by resolve_data_dir)
    tag          : label for this configuration; keep "main" for the final comparison,
                   use e.g. "lrcheck_5e-4" for tuning runs so they never mix into it.
    """
    output_dir_base = output_dir_base or RUNS_DIR
    results_path = results_path or RESULTS_PATH
    predictions_path = predictions_path or PREDICTIONS_PATH
    data_size, seed = int(data_size), int(seed)

    if skip_if_done and already_done(method_name, data_size, seed, tag, results_path):
        print(f"[skip] {method_name} data={data_size}% seed={seed} tag={tag} already logged")
        return None

    print(f"\n{'=' * 64}\n  {method_name} | data {data_size}% | seed {seed} | tag {tag} | lr {learning_rate:g}\n{'=' * 64}")

    set_seed(seed)  # BEFORE model creation

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds, test_ds = load_datasets(tokenizer, data_size)
    train_samples = len(train_ds)

    block_steps = full_steps_per_epoch()
    max_steps = MAX_EPOCHS * block_steps
    print(f"train samples: {train_samples:,} | budget: up to {max_steps:,} steps, "
          f"eval every {block_steps} steps, patience {EARLY_STOP_PATIENCE}")

    model = model_init_fn()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_params / total_params
    print(f"trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_pct:.4f}%)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    run_id = run_key(method_name, data_size, seed, tag)
    output_dir = f"{output_dir_base}/{run_id}"

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        dataloader_num_workers=2,
        max_steps=max_steps,
        learning_rate=learning_rate,
        eval_strategy="steps",       
        eval_steps=block_steps,
        save_strategy="steps",
        save_steps=block_steps,
        save_total_limit=1,
        load_best_model_at_end=True,  
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,                
        logging_strategy="steps",
        logging_steps=block_steps,
        report_to="none",
        seed=seed,
        disable_tqdm=False,
    )

    timing_cb = TimingCallback(block_steps)
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE, early_stopping_threshold=0.002), timing_cb],
    )

    t0 = time.time()
    trainer.train()
    total_time_s = time.time() - t0

    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(f"{LOGS_DIR}/{run_id}_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)

    best_val_f1 = trainer.state.best_metric
    best_step = getattr(trainer.state, "best_global_step", None)
    steps_run = trainer.state.global_step
    eval_rounds = len(timing_cb.block_times)
    avg_block_s = sum(timing_cb.block_times) / max(eval_rounds, 1)

    test_output = trainer.predict(test_ds)
    preds = np.argmax(test_output.predictions, axis=-1)
    tm = test_output.metrics
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None

    row = {
        "method": method_name, "tag": tag, "data_size": data_size, "seed": seed,
        "train_samples": train_samples,
        "accuracy": tm.get("test_accuracy"), "precision": tm.get("test_precision"),
        "recall": tm.get("test_recall"), "f1": tm.get("test_f1"),
        "trainable_params": trainable_params, "total_params": total_params,
        "trainable_pct": round(trainable_pct, 4), "learning_rate": learning_rate,
        "peak_vram_gb": round(peak_vram_gb, 3) if peak_vram_gb is not None else None,
        "avg_epoch_s": round(avg_block_s, 1), "total_time_s": round(total_time_s, 1),
        "epochs_run": eval_rounds, "steps_run": steps_run, "best_step": best_step,
        "best_val_f1": round(best_val_f1, 4) if best_val_f1 is not None else None,
        "gpu": get_gpu_name(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_result(row, results_path)
    log_predictions(method_name, tag, data_size, seed,
                    np.asarray(test_output.label_ids), preds, predictions_path)

    print(f"\nTEST -- Acc {row['accuracy']:.4f} | F1 {row['f1']:.4f} | val F1 {row['best_val_f1']} | "
          f"VRAM {row['peak_vram_gb']} GB | {total_time_s / 60:.1f} min | "
          f"{steps_run} steps ({eval_rounds} evals, best @ step {best_step})")

    del trainer, model
    shutil.rmtree(output_dir, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION (per method; cross-method comparison in 07_Final_Comparison)
# ═══════════════════════════════════════════════════════════════════════════

_FOREST = {"blue": "#2a78d6", "green": "#008300", "magenta": "#e87ba4",
           "yellow": "#eda100", "aqua": "#1baf7a", "orange": "#eb6834"}


def _load_history(method, seed, data_size=100, tag=DEFAULT_TAG):
    with open(f"{LOGS_DIR}/{run_key(method, data_size, seed, tag)}_history.json") as f:
        return json.load(f)


def plot_training_curves(method, seed, data_size=100, tag=DEFAULT_TAG):
    """Train loss + validation Accuracy/F1 vs. training step for ONE run."""
    try:
        h = _load_history(method, seed, data_size, tag)
    except FileNotFoundError:
        print(f"No saved history for {method} {data_size}% seed={seed} tag={tag} yet.")
        return
    tr = [(e["step"], e["loss"]) for e in h if "loss" in e and "step" in e]
    ev = [(e["step"], e["eval_f1"], e.get("eval_accuracy")) for e in h if "eval_f1" in e]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if tr:
        axes[0].plot(*zip(*tr), marker="o", color=_FOREST["blue"])
    axes[0].set(title=f"{method} {data_size}% (seed {seed}) -- training loss",
                xlabel="Training step", ylabel="Loss")
    axes[0].grid(alpha=0.3)

    if ev:
        steps, f1s, accs = zip(*ev)
        axes[1].plot(steps, f1s, marker="o", label="Val F1", color=_FOREST["aqua"])
        axes[1].plot(steps, accs, marker="s", label="Val Accuracy", color=_FOREST["orange"], alpha=0.7)
        b = int(np.argmax(f1s))
        axes[1].axvline(steps[b], color="gray", linestyle="--", alpha=0.6)
        axes[1].annotate("restored\ncheckpoint", xy=(steps[b], f1s[b]), xytext=(6, -4),
                         textcoords="offset points", fontsize=8, color="gray")
    axes[1].set(title=f"{method} {data_size}% (seed {seed}) -- validation",
                xlabel="Training step", ylabel="Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_seeds_overlay(method, data_size=100, seeds=None, tag=DEFAULT_TAG):
    """Validation F1 vs. step, all seeds overlaid, for one method and data size."""
    colors = list(_FOREST.values())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    n = 0
    for i, seed in enumerate(seeds or SEEDS):
        try:
            h = _load_history(method, seed, data_size, tag)
        except FileNotFoundError:
            continue
        ev = [(e["step"], e["eval_f1"]) for e in h if "eval_f1" in e]
        if ev:
            ax.plot(*zip(*ev), marker="o", label=f"seed {seed}", color=colors[i % len(colors)])
            n += 1
    if n == 0:
        print(f"No saved history for {method} {data_size}% tag={tag} yet.")
        plt.close(fig)
        return
    ax.set(xlabel="Training step", ylabel="Validation F1",
           title=f"{method} {data_size}% -- validation F1 across seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_by_size(method, sizes=None, tag=DEFAULT_TAG, predictions_path=None):
    """One confusion matrix per data size, pooled over all seeds."""
    path = predictions_path or PREDICTIONS_PATH
    _check_header(path, PRED_COLUMNS)
    df = pd.read_csv(path)
    df = df[(df["method"] == method) & (df["tag"] == tag)]
    sizes = sorted(sizes or DATA_SIZES)
    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 4))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sizes):
        sub = df[df["data_size"] == s]
        if sub.empty:
            ax.set_title(f"{method} {s}% (no data)")
            ax.axis("off")
            continue
        cm = confusion_matrix(sub["true_label"], sub["predicted_label"], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels([ID2LABEL[0], ID2LABEL[1]])
        ax.set_yticks([0, 1]); ax.set_yticklabels([ID2LABEL[0], ID2LABEL[1]])
        ax.set(xlabel="Predicted", ylabel="True",
               title=f"{method} {s}% ({sub['seed'].nunique()} seeds pooled)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.show()


def plot_method_summary(method, tag=DEFAULT_TAG, results_path=None):
    """Mean +- std over seeds, by data size: test/val F1, generalization gap, time."""
    df = load_results(results_path)
    sub = df[(df["method"] == method) & (df["tag"] == tag)]
    if sub.empty:
        print(f"No results logged yet for {method} (tag={tag}).")
        return
    g = sub.groupby("data_size")
    sizes = sorted(sub["data_size"].unique())
    f1m, f1s = g["f1"].mean().reindex(sizes), g["f1"].std().reindex(sizes)
    vm, vs = g["best_val_f1"].mean().reindex(sizes), g["best_val_f1"].std().reindex(sizes)
    gap = (vm - f1m)
    tm = (g["total_time_s"].mean() / 60).reindex(sizes)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].errorbar(sizes, f1m, yerr=f1s, marker="o", capsize=3, label="Test F1", color=_FOREST["blue"])
    axes[0].errorbar(sizes, vm, yerr=vs, marker="s", linestyle="--", capsize=3,
                     label="Val F1 (best)", color=_FOREST["yellow"])
    axes[0].set_xscale("log"); axes[0].set_xticks(sizes); axes[0].set_xticklabels([f"{s}%" for s in sizes])
    axes[0].set(xlabel="Training data", ylabel="F1", title=f"{method} -- F1 vs. data size")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar([f"{s}%" for s in sizes], gap, color=_FOREST["orange"])
    axes[1].set(ylabel="Val F1 - Test F1", title="Generalization gap")
    axes[1].axhline(0, color="black", linewidth=0.8); axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar([f"{s}%" for s in sizes], tm, color=_FOREST["aqua"])
    axes[2].set(ylabel="Minutes per run", title="Training time per run")
    axes[2].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    cols = ["data_size", "seed", "best_val_f1", "accuracy", "f1", "trainable_pct",
            "peak_vram_gb", "total_time_s", "epochs_run", "steps_run"]
    print(sub.sort_values(["data_size", "seed"])[cols].to_string(index=False))
