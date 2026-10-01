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
import time
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ═══════════════════════════════════════════════════════════════════════════
# FIXED CONFIGURATION -- identical across all six methods.
# Do not override these per-notebook; only learning_rate and the model_init
# function differ between methods.
# ═══════════════════════════════════════════════════════════════════════════

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 128
BATCH_SIZE = 16
MAX_EPOCHS = 15
EARLY_STOP_PATIENCE = 3
LABEL2ID = {"fact": 0, "hoax": 1}
ID2LABEL = {0: "fact", 1: "hoax"}
SEEDS = [42, 123, 456]

# ── Paths -- adjust these to match your Drive layout ────────────────────────
# DATA_DIR = "/content/drive/MyDrive/thesis/data"          # expects train.csv, val.csv, test.csv
DATA_DIR = "/datasets"
RESULTS_PATH = "/results.csv"
PREDICTIONS_PATH = "/predictions.csv"
RUNS_DIR = "/content/runs"                                 # local scratch, not Drive (checkpoints are large)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING -- identical split, identical tokenization, every method/seed.
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(tokenizer):
    """Load the fixed train/val/test CSVs and tokenize consistently.

    Expects each CSV to have 'title' and 'label' columns, with label already
    encoded as 0 (fact) / 1 (hoax). Adjust the label mapping below if your
    CSVs still store string labels.
    """
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert "title" in df.columns and "label" in df.columns, \
            f"{name}.csv must have 'title' and 'label' columns, got {list(df.columns)}"

    # If labels are still strings ("fact"/"hoax"), map them to ints here.
    for df in (train_df, val_df, test_df):
        if df["label"].dtype == object:
            df["label"] = df["label"].map(LABEL2ID)

    def to_hf_dataset(df):
        return Dataset.from_pandas(df[["title", "label"]].reset_index(drop=True))

    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)
    test_ds = to_hf_dataset(test_df)

    def tokenize_fn(batch):
        return tokenizer(
            batch["title"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

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


def check_train_test_leakage():
    """One-time sanity check: exact-duplicate titles shared between train and
    test would inflate every method's accuracy by the same amount. Run this
    once after the split is created, not per-method/per-seed."""
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    overlap = set(train_df["title"]) & set(test_df["title"])
    print(f"Exact duplicate titles between train/test: {len(overlap)}")
    if len(overlap) > 0:
        print("Sample overlapping titles:")
        for t in list(overlap)[:5]:
            print(" -", t)
    return overlap


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
    )

    timing_cb = TimingCallback()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE),
            timing_cb,
        ],
    )

    t0 = time.time()
    trainer.train()
    total_time_s = time.time() - t0

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
    log_predictions(method_name, seed, test_ds["labels"].numpy(), preds, predictions_path)

    print(f"\nTest results -- Acc: {row['accuracy']:.4f} | F1: {row['f1']:.4f} | "
          f"Peak VRAM: {row['peak_vram_gb']} GB | "
          f"Total time: {total_time_s/60:.1f} min | Epochs: {epochs_run}")

    return row
