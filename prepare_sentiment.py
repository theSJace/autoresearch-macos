"""
One-time data preparation for the financial sentiment task.
Downloads FinancialPhraseBank from HuggingFace, trains the sentiment tokenizer
(reuses the tokenizer trained by prepare.py), and provides the dataloader and
evaluation function used by train_sentiment.py.

Usage:
    uv run prepare_sentiment.py

Prerequisites:
    uv run prepare.py --num-shards 1   # trains the tokenizer (needed once)

Data and model are cached in ~/.cache/autoresearch/sentiment/.
"""

import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 512      # max token length per sample (shorter than LM task)
TIME_BUDGET = 300      # training time budget in seconds (5 minutes)
EVAL_SAMPLES = None    # use all val samples for evaluation

SENTIMENT_CACHE_DIR = Path.home() / ".cache" / "autoresearch" / "sentiment"
TOKENIZER_DIR = Path.home() / ".cache" / "autoresearch" / "tokenizer"

# Label mapping for FinancialPhraseBank: 0=negative, 1=neutral, 2=positive
# We binarize: positive(2) → 1.0, negative(0) → 0.0 (neutral filtered out)
BINARY_LABELS = {0: 0.0, 2: 1.0}

# HuggingFace dataset: nickmuchi/financial-classification (financial_phrasebank config)
HF_DATASET = "nickmuchi/financial-classification"
HF_CONFIG = "financial_phrasebank"

# ---------------------------------------------------------------------------
# Tokenizer (standalone — does not import from prepare.py to avoid macOS check)
# ---------------------------------------------------------------------------

BOS_TOKEN = "<|reserved_0|>"


class SentimentTokenizer:
    """Minimal tokenizer wrapper. Loads the BPE tokenizer trained by prepare.py."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_cache(cls, tokenizer_dir=None):
        if tokenizer_dir is None:
            tokenizer_dir = TOKENIZER_DIR
        tok_path = tokenizer_dir / "tokenizer.pkl"
        if not tok_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tok_path}.\n"
                "Run first: uv run prepare.py --num-shards 1"
            )
        with open(tok_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def encode(self, text):
        """Encode text, prepending BOS token."""
        ids = self.enc.encode_ordinary(text)
        return [self.bos_id] + ids

    def get_vocab_size(self):
        return self.enc.n_vocab


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _get_parquet_url():
    """Resolve the HuggingFace parquet URL via the datasets-server API."""
    api_url = (
        f"https://datasets-server.huggingface.co/parquet"
        f"?dataset={HF_DATASET}&config={HF_CONFIG}&split=train"
    )
    try:
        resp = requests.get(api_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        files = data.get("parquet_files", [])
        if files:
            return files[0]["url"]
    except Exception:
        pass
    return None


def download_dataset():
    """Download FinancialPhraseBank parquet from HuggingFace. Caches locally."""
    SENTIMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = SENTIMENT_CACHE_DIR / "financial_phrasebank.parquet"

    if parquet_path.exists():
        print(f"Dataset: already downloaded at {parquet_path}")
        return parquet_path

    # Try to resolve URL via HF datasets server API
    url = _get_parquet_url()

    # Fallback URLs (standard HF CDN patterns)
    fallback_urls = [
        f"https://huggingface.co/datasets/{HF_DATASET}/resolve/refs%2Fconvert%2Fparquet/{HF_CONFIG}/train/0000.parquet",
        f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/data/train-00000-of-00001.parquet",
    ]
    urls_to_try = ([url] if url else []) + fallback_urls

    print(f"Dataset: downloading {HF_DATASET} ({HF_CONFIG})...")
    last_error = None
    for attempt_url in urls_to_try:
        try:
            resp = requests.get(attempt_url, stream=True, timeout=30)
            resp.raise_for_status()
            temp_path = str(parquet_path) + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, parquet_path)
            print(f"Dataset: saved to {parquet_path}")
            return parquet_path
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to download {HF_DATASET}. Last error: {last_error}\n"
        "Check your internet connection or download manually:\n"
        f"  https://huggingface.co/datasets/{HF_DATASET}"
    )


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_split(parquet_path=None, val_fraction=0.2, seed=42):
    """
    Load FinancialPhraseBank, filter neutral labels, binarize, split train/val.

    Returns (train_df, val_df) with columns: sentence, binary_label
    """
    if parquet_path is None:
        parquet_path = SENTIMENT_CACHE_DIR / "financial_phrasebank.parquet"

    df = pd.read_parquet(parquet_path)

    # The dataset has a 'sentence' column and a 'label' column (0/1/2)
    # Some versions use different column names — normalise
    if "sentence" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "sentence"})
    if "label" not in df.columns and "labels" in df.columns:
        df = df.rename(columns={"labels": "label"})

    # Filter to positive (2) and negative (0) only
    df = df[df["label"].isin(BINARY_LABELS)].copy()
    df["binary_label"] = df["label"].map(BINARY_LABELS)

    # Shuffle and split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Dataset: {len(train_df)} train / {len(val_df)} val samples")
    pos_train = int(train_df["binary_label"].sum())
    print(f"  Train: {pos_train} positive ({pos_train/len(train_df):.1%}), "
          f"{len(train_df)-pos_train} negative")
    pos_val = int(val_df["binary_label"].sum())
    print(f"  Val:   {pos_val} positive ({pos_val/len(val_df):.1%}), "
          f"{len(val_df)-pos_val} negative")

    return train_df, val_df


class SentimentDataset(Dataset):
    """PyTorch Dataset for financial sentiment classification."""

    def __init__(self, df, tokenizer, max_seq_len=MAX_SEQ_LEN):
        self.tokens = []
        self.labels = []
        for _, row in df.iterrows():
            ids = tokenizer.encode(str(row["sentence"]))[:max_seq_len]
            self.tokens.append(ids)
            self.labels.append(float(row["binary_label"]))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


def _collate_fn(batch):
    """Pad a batch of variable-length sequences. Returns (tokens, labels, mask)."""
    tokens_list, labels = zip(*batch)
    max_len = max(len(t) for t in tokens_list)
    B = len(tokens_list)

    padded = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, t in enumerate(tokens_list):
        padded[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        mask[i, : len(t)] = True

    return padded, torch.tensor(labels, dtype=torch.float32), mask


def make_sentiment_dataloader(tokenizer, split="train", batch_size=32):
    """
    Returns a DataLoader for the sentiment task.
    Yields (tokens, labels, mask) batches.
    """
    parquet_path = SENTIMENT_CACHE_DIR / "financial_phrasebank.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Run: uv run prepare_sentiment.py"
        )
    train_df, val_df = load_and_split(parquet_path)
    df = train_df if split == "train" else val_df
    dataset = SentimentDataset(df, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=_collate_fn,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_auc(probs, labels):
    """Compute AUC-ROC using numpy trapezoidal integration. No sklearn needed."""
    probs = np.array(probs, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)

    sorted_idx = np.argsort(-probs)
    labels_sorted = labels[sorted_idx]

    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5  # degenerate case

    tp, fp = 0.0, 0.0
    tpr_pts = [0.0]
    fpr_pts = [0.0]
    for lbl in labels_sorted:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr_pts.append(tp / pos)
        fpr_pts.append(fp / neg)
    tpr_pts.append(1.0)
    fpr_pts.append(1.0)

    return float(np.trapz(tpr_pts, fpr_pts))


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, device, batch_size=64):
    """
    Evaluate model on the validation split.
    Returns (val_accuracy, val_auc).
    """
    parquet_path = SENTIMENT_CACHE_DIR / "financial_phrasebank.parquet"
    _, val_df = load_and_split(parquet_path)
    dataset = SentimentDataset(val_df, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
    )

    model.eval()
    all_probs = []
    all_labels = []

    for tokens, labels, mask in loader:
        tokens = tokens.to(device)
        mask = mask.to(device)
        cls_logits = model(tokens, mask=mask)
        probs = torch.sigmoid(cls_logits.float()).cpu().numpy().tolist()
        all_probs.extend(probs)
        all_labels.extend(labels.tolist())

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    accuracy = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
    auc = compute_auc(all_probs, all_labels)
    return float(accuracy), float(auc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cache directory: {SENTIMENT_CACHE_DIR}")
    print()

    # Step 1: Download dataset
    parquet_path = download_dataset()
    print()

    # Step 2: Load and split
    train_df, val_df = load_and_split(parquet_path)
    print()

    # Step 3: Load tokenizer (requires prepare.py to have been run)
    tokenizer = SentimentTokenizer.from_cache()
    print(f"Tokenizer: vocab_size={tokenizer.get_vocab_size():,}")

    # Step 4: Smoke-test the dataloader
    loader = make_sentiment_dataloader(tokenizer, split="train", batch_size=4)
    tokens, labels, mask = next(iter(loader))
    print(f"\nSample batch:")
    print(f"  tokens shape: {tokens.shape}")
    print(f"  labels:       {labels.tolist()}")
    print(f"  mask shape:   {mask.shape}")
    print(f"  sample text:  '{train_df.iloc[0]['sentence'][:80]}'")
    print(f"  sample label: {train_df.iloc[0]['binary_label']} "
          f"({'bullish' if train_df.iloc[0]['binary_label'] == 1.0 else 'bearish'})")
    print()
    print("Done! Ready to run: uv run train_sentiment.py")
