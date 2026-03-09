"""
Score financial text for bullish/bearish sentiment.

Loads the model saved by train_sentiment.py and scores any financial text:
earnings call transcripts, analyst commentary, guidance statements, news.

Usage:
    echo "Revenue grew 28% year-over-year with expanding margins." | uv run score_transcript.py
    uv run score_transcript.py "Management reaffirmed full-year guidance..."
    uv run score_transcript.py transcript.txt          # file path

Output:
    Score: 0.7823
    Signal: BULLISH (78.2% confidence)

Long transcripts are chunked automatically via sliding window.
The final score is the mean probability across all chunks.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_sentiment import (
    SentimentTokenizer, SENTIMENT_CACHE_DIR, MAX_SEQ_LEN,
)

# ---------------------------------------------------------------------------
# Model class (must match train_sentiment.py — keep in sync)
# Imported directly so score_transcript.py works even if train_sentiment.py
# is modified by the agent (as long as the class interface is stable).
# ---------------------------------------------------------------------------

from train_sentiment import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Sliding-window chunking for long transcripts
# ---------------------------------------------------------------------------

CHUNK_SIZE = MAX_SEQ_LEN - 32   # leave a small margin for safety
CHUNK_STRIDE = CHUNK_SIZE // 4  # 75% overlap gives stable ensemble predictions


def chunk_tokens(token_ids, chunk_size=CHUNK_SIZE, stride=CHUNK_STRIDE):
    """
    Split a token sequence into overlapping chunks.
    Every chunk starts with the BOS token from the original sequence.
    Returns list of token lists (each length <= chunk_size).
    """
    if len(token_ids) <= chunk_size:
        return [token_ids]

    chunks = []
    bos = token_ids[0]  # BOS token is always first (encode() prepends it)
    content = token_ids[1:]  # strip BOS for chunking

    for start in range(0, len(content), stride):
        chunk_content = content[start : start + chunk_size - 1]  # -1 to make room for BOS
        chunks.append([bos] + chunk_content)
        if start + chunk_size - 1 >= len(content):
            break

    return chunks if chunks else [token_ids[:chunk_size]]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path=None, device=None):
    """Load the trained sentiment model from checkpoint."""
    if model_path is None:
        model_path = SENTIMENT_CACHE_DIR / "model.pt"

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}.\n"
            "Train first: uv run train_sentiment.py"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]
    config = GPTConfig(**config_dict)

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    accuracy = checkpoint.get("val_accuracy", None)
    auc = checkpoint.get("val_auc", None)
    if accuracy is not None:
        print(f"Loaded model: val_accuracy={accuracy:.4f}, val_auc={auc:.4f}")

    return model, device


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_text(text, model, tokenizer, device):
    """
    Score financial text. Returns P(bullish) in [0.0, 1.0].

    For short texts: single forward pass.
    For long texts: sliding-window ensemble (mean probability).
    """
    token_ids = tokenizer.encode(text)

    if not token_ids:
        return 0.5  # empty input → neutral

    chunks = chunk_tokens(token_ids)
    probs = []

    for chunk in chunks:
        t = torch.tensor([chunk], dtype=torch.long, device=device)
        logit = model(t)                          # (1,) cls logit
        prob = torch.sigmoid(logit).item()
        probs.append(prob)

    return sum(probs) / len(probs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # Determine input source
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            with open(arg, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            print(f"Scoring file: {arg} ({len(text):,} chars)", file=sys.stderr)
        else:
            text = arg
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print(__doc__)
        sys.exit(0)

    text = text.strip()
    if not text:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer and model
    tokenizer = SentimentTokenizer.from_cache()
    model, device = load_model()

    # Score
    prob = score_text(text, model, tokenizer, device)

    # Output
    label = "BULLISH" if prob > 0.5 else "BEARISH"
    confidence = prob if prob > 0.5 else 1.0 - prob

    print(f"Score:  {prob:.4f}")
    print(f"Signal: {label} ({confidence:.1%} confidence)")

    # Bonus: if text is long enough, show per-chunk breakdown
    token_ids = tokenizer.encode(text)
    chunks = chunk_tokens(token_ids)
    if len(chunks) > 1:
        print(f"\n{len(chunks)} chunks analysed (sliding window, {CHUNK_SIZE} tokens, stride {CHUNK_STRIDE}):")
        for i, chunk in enumerate(chunks):
            t = torch.tensor([chunk], dtype=torch.long, device=device)
            with torch.no_grad():
                logit = model(t)
            p = torch.sigmoid(logit).item()
            lbl = "↑" if p > 0.5 else "↓"
            print(f"  chunk {i+1:2d}: {p:.4f} {lbl}")


if __name__ == "__main__":
    main()
