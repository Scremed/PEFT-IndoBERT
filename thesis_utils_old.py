# -*- coding: utf-8 -*-
"""
thesis_utils.py
================
Shared utilities for the IndoBERT PEFT comparison experiments
(FFT, LoRA, BitFit, IA3, Houlsby, Pfeiffer).

Import this SAME module in every method notebook. Do not copy-paste these
functions into individual notebooks -- that's exactly the kind of drift
(different max_length, different eval logic, forgotten seeding) that would
quietly bias one method's results relative to another.

Upload this file once to your Drive code folder and import it from every
notebook:

    import sys
    sys.path.append('/content/drive/MyDrive/thesis/code')
    from thesis_utils import *
"""

import os
import csv
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# ═══════════════════════════════════════════════════════════════════════════
# FIXED CONFIGURATION -- identical across all six methods.
# Do not override these per-notebook; only learning_rate and the model_init
# function differ between methods.
# ═══════════════════════════════════════════════════════════════════════════

MODEL_NAME = "indobenchmark/indobert-base-p2"
MAX_LENGTH = 128
BATCH_SIZE = 16
MAX_EPOCHS = 15
EARLY_STOP_PATIENCE = 3
LABEL2ID = {"fact": 0, "hoax": 1}
ID2LABEL = {0: "fact", 1: "hoax"}
SEEDS = [42, 123, 456]

# ── Paths -- adjust these to match your Drive layout ────────────────────────
#DATA_DIR = "/content/drive/MyDrive/thesis/data"          # expects train.csv, val.csv, test.csv
#RESULTS_PATH = "/content/drive/MyDrive/thesis/results.csv"
#PREDICTIONS_PATH = "/content/drive/MyDrive/thesis/predictions.csv"
#LOGS_DIR = "/content/drive/MyDrive/thesis/logs"           # per-run training history, for plotting
#RUNS_DIR = "/content/runs"                                 # local scratch, not Drive (checkpoints are large)

# New Datasets
DATA_DIR = "/content/drive/MyDrive/thesis/data/split_70_10_20"          # expects train.csv, val.csv, test.csv
RESULTS_PATH = "/content/drive/MyDrive/thesis/new_results.csv"
PREDICTIONS_PATH = "/content/drive/MyDrive/thesis/new_predictions.csv"
LOGS_DIR = "/content/drive/MyDrive/thesis/logs/new"           # per-run training history, for plotting
RUNS_DIR = "/content/runs"                                 # local scratch, not Drive (checkpoints are large)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING -- identical split, identical tokenization, every method/seed.
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(tokenizer):
    """Load the fixed train/val/test CSVs and tokenize consistently.

    Expects each CSV to have 'berita' and 'label' columns, with label already
    encoded as 0 (fact) / 1 (hoax). Adjust the label mapping below if your
    CSVs still store string labels.
    """
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert "berita" in df.columns and "label" in df.columns, \
            f"{name}.csv must have 'berita' and 'label' columns, got {list(df.columns)}"

    # If labels are still strings ("fact"/"hoax"), map them to ints here.
    for df in (train_df, val_df, test_df):
        if df["label"].dtype == object:
            df["label"] = df["label"].map(LABEL2ID)

    def to_hf_dataset(df):
        return Dataset.from_pandas(df[["berita", "label"]].reset_index(drop=True))

    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)
    test_ds = to_hf_dataset(test_df)

    def tokenize_fn(batch):
        return tokenizer(batch["berita"], truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    columns = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)
    test_ds.set_format(type="torch", columns=columns)

    return train_ds, val_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ═══════════════════════════════════════════════════════════════════════════
# TIMING -- exact per-epoch wall-clock time, independent of Trainer's own logs
# ═══════════════════════════════════════════════════════════════════════════

class TimingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_times = []
        self._epoch_start = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_start is not None:
            self.epoch_times.append(time.time() - self._epoch_start)


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING -- append immediately after every (method, seed) cell finishes.
# Never batch results in memory until "the end of the notebook" -- a
# disconnect should cost you the current cell, not the whole session.
# ═══════════════════════════════════════════════════════════════════════════

def get_gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def log_result(row: dict, results_path: str = RESULTS_PATH):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    exists = os.path.exists(results_path)
    with open(results_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def log_predictions(method, seed, true_labels, preds, predictions_path: str = PREDICTIONS_PATH):
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    exists = os.path.exists(predictions_path)
    with open(predictions_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["method", "seed", "index", "true_label", "predicted_label"])
        for i, (t, p) in enumerate(zip(true_labels, preds)):
            w.writerow([method, seed, i, int(t), int(p)])


def already_done(method, seed, results_path: str = RESULTS_PATH):
    """Check before starting a run -- lets you re-run a notebook after a
    disconnect without duplicating (method, seed) rows already logged."""
    if not os.path.exists(results_path):
        return False
    df = pd.read_csv(results_path)
    if df.empty:
        return False
    return ((df["method"] == method) & (df["seed"] == seed)).any()


# ═══════════════════════════════════════════════════════════════════════════
# CORE TRAINING LOOP -- identical for every method. Only model_init_fn and
# learning_rate change between FFT / LoRA / BitFit / IA3 / Houlsby / Pfeiffer.
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(method_name, seed, model_init_fn, learning_rate,
                    output_dir_base=RUNS_DIR,
                    results_path=RESULTS_PATH,
                    predictions_path=PREDICTIONS_PATH,
                    skip_if_done=True):
    """
    method_name  : e.g. "FFT", "LoRA", "BitFit"
    seed         : int, one of SEEDS
    model_init_fn: zero-arg callable returning a FRESH model. Must be called
                   AFTER set_seed() (this function handles that ordering),
                   so the classification head's random init is also seeded.
    learning_rate: method-specific LR (FFT=2e-5, LoRA=1e-4, BitFit=1e-3, ...)
    """
    if skip_if_done and already_done(method_name, seed, results_path):
        print(f"[skip] {method_name} seed={seed} already logged in {results_path}")
        return None

    print(f"\n{'='*60}\n  METHOD: {method_name} | SEED: {seed}\n{'='*60}")

    # Seed BEFORE model instantiation -- controls the classifier head's
    # random init, not just dataloader order.
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds, test_ds = load_datasets(tokenizer)

    model = model_init_fn()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_params / total_params
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({trainable_pct:.4f}%)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output_dir = f"{output_dir_base}/{method_name}_seed{seed}"

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=MAX_EPOCHS,
        learning_rate=learning_rate,
        eval_strategy="epoch",     # older transformers (<4.41): use evaluation_strategy=
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,                # fp32 -- current decision, keep identical across methods
        logging_strategy="epoch",
        report_to="none",
        seed=seed,
        disable_tqdm=False,
        dataloader_num_workers=2,
    )

    timing_cb = TimingCallback()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE),
            timing_cb,
        ],
    )

    t0 = time.time()
    trainer.train()
    total_time_s = time.time() - t0

    # Save full epoch-by-epoch history now, while the Trainer object still
    # exists -- plotting functions read this back later without needing to
    # keep the Trainer (or even this session) alive.
    os.makedirs(LOGS_DIR, exist_ok=True)
    history_path = f"{LOGS_DIR}/{method_name}_{seed}_history.json"
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f)

    # load_best_model_at_end=True already restored the best-val-F1 checkpoint
    # -- do not evaluate the last epoch's weights instead.
    best_val_f1 = trainer.state.best_metric
    epochs_run = len(timing_cb.epoch_times)
    avg_epoch_s = sum(timing_cb.epoch_times) / max(epochs_run, 1)

    test_output = trainer.predict(test_ds)
    preds = np.argmax(test_output.predictions, axis=-1)
    test_metrics = test_output.metrics

    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None

    row = {
        "method": method_name,
        "seed": seed,
        "accuracy": test_metrics.get("test_accuracy"),
        "precision": test_metrics.get("test_precision"),
        "recall": test_metrics.get("test_recall"),
        "f1": test_metrics.get("test_f1"),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(trainable_pct, 4),
        "learning_rate": learning_rate,
        "peak_vram_gb": round(peak_vram_gb, 3) if peak_vram_gb is not None else None,
        "avg_epoch_s": round(avg_epoch_s, 1),
        "total_time_s": round(total_time_s, 1),
        "epochs_run": epochs_run,
        "best_val_f1": round(best_val_f1, 4) if best_val_f1 is not None else None,
        "gpu": get_gpu_name(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log_result(row, results_path)
    log_predictions(method_name, seed, test_ds["labels"], preds, predictions_path)

    print(f"\nTest results -- Acc: {row['accuracy']:.4f} | F1: {row['f1']:.4f} | "
          f"Peak VRAM: {row['peak_vram_gb']} GB | "
          f"Total time: {total_time_s/60:.1f} min | Epochs: {epochs_run}")

    return row


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION -- call these after run_experiment() finishes for a method.
# All read back from the CSVs/JSON already written by run_experiment, so
# they work even in a fresh session (e.g. re-opening the notebook later)
# as long as Drive is mounted and that method's runs have been logged.
# ═══════════════════════════════════════════════════════════════════════════

_FOREST = {
    "blue": "#2a78d6", "green": "#008300", "magenta": "#e87ba4",
    "yellow": "#eda100", "aqua": "#1baf7a", "orange": "#eb6834",
}


def _load_history(method, seed, logs_dir=LOGS_DIR):
    path = f"{logs_dir}/{method}_{seed}_history.json"
    with open(path) as f:
        return json.load(f)


def plot_training_curves(method, seed, logs_dir=LOGS_DIR):
    """
    Plot training loss vs validation loss across epochs.

    Useful for detecting:
      - Overfitting: train loss keeps decreasing while val loss increases
      - Underfitting: both losses remain high / decrease slowly
      - Good generalization: both losses decrease and remain relatively close

    Also marks the epoch with the best validation F1, which is the
    checkpoint restored by load_best_model_at_end=True.
    """
    try:
        history = _load_history(method, seed, logs_dir)
    except FileNotFoundError:
        print(f"No saved history for {method} seed={seed} yet -- run it first.")
        return

    # Training loss is logged at the end of each epoch
    train_epochs = [e["epoch"] for e in history if "loss" in e]
    train_loss = [e["loss"] for e in history if "loss" in e]

    # Validation loss/metrics are logged during evaluation
    eval_entries = [e for e in history if "eval_loss" in e]

    eval_epochs = [e["epoch"] for e in eval_entries]
    eval_loss = [e["eval_loss"] for e in eval_entries]
    eval_f1 = [e.get("eval_f1") for e in eval_entries]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Training loss
    ax.plot(
        train_epochs,
        train_loss,
        marker="o",
        label="Training Loss",
        color=_FOREST["blue"]
    )

    # Validation loss
    ax.plot(
        eval_epochs,
        eval_loss,
        marker="s",
        label="Validation Loss",
        color=_FOREST["orange"]
    )

    # Mark best validation F1 checkpoint
    valid_f1 = [
        (epoch, f1)
        for epoch, f1 in zip(eval_epochs, eval_f1)
        if f1 is not None
    ]

    if valid_f1:
        best_epoch, best_f1 = max(valid_f1, key=lambda x: x[1])

        ax.axvline(
            best_epoch,
            color="gray",
            linestyle="--",
            alpha=0.6,
            label=f"Best Val F1 (epoch {best_epoch:g})"
        )

    ax.set_title(f"{method} (seed {seed}) — Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_seeds_overlay(method, seeds=SEEDS, logs_dir=LOGS_DIR):
    """Overlay validation F1 curves across all seeds for one method --
    a quick visual check for how consistent (or noisy) training is across
    seeds, before trusting the mean+-std in the summary table."""
    colors = list(_FOREST.values())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    plotted = 0
    for i, seed in enumerate(seeds):
        try:
            history = _load_history(method, seed, logs_dir)
        except FileNotFoundError:
            continue
        eval_epochs = [e["epoch"] for e in history if "eval_f1" in e]
        eval_f1 = [e["eval_f1"] for e in history if "eval_f1" in e]
        ax.plot(eval_epochs, eval_f1, marker="o", label=f"seed {seed}",
                 color=colors[i % len(colors)])
        plotted += 1
    if plotted == 0:
        print(f"No saved history for {method} yet -- run it first.")
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title(f"{method} -- validation F1 across seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(method, seed, predictions_path=PREDICTIONS_PATH):
    """Confusion matrix on the test set for ONE (method, seed) run."""
    df = pd.read_csv(predictions_path)
    subset = df[(df["method"] == method) & (df["seed"] == seed)]
    if subset.empty:
        print(f"No predictions logged yet for {method} seed={seed}.")
        return None

    cm = confusion_matrix(subset["true_label"], subset["predicted_label"])
    labels = [ID2LABEL[0], ID2LABEL[1]]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{method} (seed {seed})")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.show()
    return cm


def plot_method_summary(method, results_path=RESULTS_PATH):
    """Bar chart (mean +- std across seeds) of validation F1, test metrics,
    VRAM, and training time for one method -- plus the raw per-seed table.
    Val F1 is included alongside the test metrics specifically so a gap
    between them (like the one seen for LoRA/BitFit in the pilot) is
    visible immediately, not just discoverable by cross-referencing columns."""
    df = pd.read_csv(results_path)
    subset = df[df["method"] == method]
    if subset.empty:
        print(f"No results logged yet for {method}.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    perf_cols = ["best_val_f1", "accuracy", "precision", "recall", "f1"]
    perf_labels = ["Val F1\n(best)", "Test\nAcc", "Test\nPrec", "Test\nRec", "Test\nF1"]
    means = [subset[c].mean() for c in perf_cols]
    stds = [subset[c].std() for c in perf_cols]
    bar_colors = [_FOREST["yellow"]] + [_FOREST["blue"]] * 4  # val F1 stands out from test metrics
    axes[0].bar(perf_labels, means, yerr=stds, capsize=4, color=bar_colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"{method} -- performance (n={len(subset)} seeds)")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(["Peak VRAM (GB)"], [subset["peak_vram_gb"].mean()],
                 yerr=[subset["peak_vram_gb"].std()], capsize=4, color=_FOREST["orange"])
    axes[1].set_title("Peak VRAM")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(["Total time (min)"], [subset["total_time_s"].mean() / 60],
                 yerr=[subset["total_time_s"].std() / 60], capsize=4, color=_FOREST["aqua"])
    axes[2].set_title("Total training time")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    display_cols = ["seed"] + perf_cols + ["trainable_pct", "peak_vram_gb", "total_time_s", "epochs_run"]
    print(subset[display_cols].to_string(index=False))
