"""
Cron-triggered overnight training runner.

Reads the idea saved by the Telegram bot, patches train_sentiment.py with any
recognised hyperparameter assignments, runs the 5-minute training sprint, then
messages you the results via Telegram.

Usage:
    uv run schedule_run.py              # run the pending scheduled idea
    uv run schedule_run.py --setup      # print the crontab line to add
    uv run schedule_run.py --now "LM_LOSS_WEIGHT=0.1 DEPTH=6"  # run ad-hoc

Typical cron setup (runs at 11pm every night):
    0 23 * * * cd /path/to/autoresearch-macos && uv run schedule_run.py >> cron.log 2>&1
    (see --setup flag for an auto-generated line with correct paths)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_DIR / "train_sentiment.py"
RUN_LOG = REPO_DIR / "run.log"

from prepare_sentiment import SENTIMENT_CACHE_DIR
SCHEDULED_IDEA_PATH = SENTIMENT_CACHE_DIR / "scheduled_idea.json"

# ---------------------------------------------------------------------------
# Hyperparameter patching
# ---------------------------------------------------------------------------

# Maps UPPER_CASE name → (line-matching regex, Python type)
# The regex captures the assignment prefix so we can replace just the value.
KNOWN_HYPERPARAMS: dict[str, tuple[str, type]] = {
    "DEPTH":             (r"^(DEPTH\s*=\s*)\S+",          int),
    "ASPECT_RATIO":      (r"^(ASPECT_RATIO\s*=\s*)\S+",   int),
    "HEAD_DIM":          (r"^(HEAD_DIM\s*=\s*)\S+",        int),
    "CLS_DROPOUT":       (r"^(CLS_DROPOUT\s*=\s*)\S+",    float),
    "LM_LOSS_WEIGHT":    (r"^(LM_LOSS_WEIGHT\s*=\s*)\S+", float),
    "DEVICE_BATCH_SIZE": (r"^(DEVICE_BATCH_SIZE\s*=\s*)\S+", int),
    "EMBEDDING_LR":      (r"^(EMBEDDING_LR\s*=\s*)\S+",   float),
    "UNEMBEDDING_LR":    (r"^(UNEMBEDDING_LR\s*=\s*)\S+", float),
    "MATRIX_LR":         (r"^(MATRIX_LR\s*=\s*)\S+",      float),
    "SCALAR_LR":         (r"^(SCALAR_LR\s*=\s*)\S+",      float),
    "WEIGHT_DECAY":      (r"^(WEIGHT_DECAY\s*=\s*)\S+",   float),
    "WARMUP_RATIO":      (r"^(WARMUP_RATIO\s*=\s*)\S+",   float),
    "WARMDOWN_RATIO":    (r"^(WARMDOWN_RATIO\s*=\s*)\S+", float),
    "FINAL_LR_FRAC":     (r"^(FINAL_LR_FRAC\s*=\s*)\S+",  float),
    "WINDOW_PATTERN":    (r'^(WINDOW_PATTERN\s*=\s*)"[^"]*"', str),
}


def parse_hyperparams(idea_text: str) -> dict:
    """Extract `NAME=value` pairs from free-form idea text."""
    changes = {}
    for m in re.finditer(r'(\w+)\s*=\s*(["\']?[\w.]+["\']?)', idea_text, re.IGNORECASE):
        name = m.group(1).upper()
        raw_val = m.group(2).strip("\"'")
        if name in KNOWN_HYPERPARAMS:
            _, type_fn = KNOWN_HYPERPARAMS[name]
            try:
                changes[name] = type_fn(raw_val)
            except ValueError:
                print(f"Warning: could not parse {name}={raw_val!r} — skipping")
    return changes


def apply_hyperparams(changes: dict) -> list[str]:
    """
    Patch train_sentiment.py in-place with the given hyperparameter values.
    Returns list of 'NAME=value' strings that were successfully applied.
    """
    if not changes:
        return []

    code = TRAIN_SCRIPT.read_text()
    applied = []

    for name, value in changes.items():
        pattern_str, type_fn = KNOWN_HYPERPARAMS[name]
        if type_fn == str:
            replacement = f'{name} = "{value}"'
        elif type_fn == int:
            replacement = f"{name} = {int(value)}"
        else:
            replacement = f"{name} = {float(value)}"

        new_code, n = re.subn(pattern_str, replacement, code, flags=re.MULTILINE)
        if n > 0:
            code = new_code
            applied.append(f"{name}={value}")
        else:
            print(f"Warning: could not find {name} in train_sentiment.py — skipping")

    if applied:
        TRAIN_SCRIPT.write_text(code)
        print(f"Patched train_sentiment.py: {', '.join(applied)}")

    return applied


def revert_train_script():
    """Reset train_sentiment.py to HEAD (undo any patches)."""
    subprocess.run(
        ["git", "checkout", "HEAD", "--", "train_sentiment.py"],
        cwd=REPO_DIR, check=False, capture_output=True,
    )
    print("Reverted train_sentiment.py to HEAD.")


def git_commit_changes(message: str):
    subprocess.run(["git", "add", "train_sentiment.py"], cwd=REPO_DIR, check=False)
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=REPO_DIR, capture_output=True, text=True,
    )
    if result.returncode == 0:
        short_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()
        print(f"Committed: {short_hash} — {message}")
        return short_hash
    return ""


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_training(timeout_secs: int = 700) -> bool:
    """Run train_sentiment.py, capturing output to run.log. Returns True on success."""
    print(f"Running: uv run train_sentiment.py  (timeout {timeout_secs}s)")
    try:
        with open(RUN_LOG, "w") as log:
            proc = subprocess.run(
                ["uv", "run", "train_sentiment.py"],
                cwd=REPO_DIR,
                stdout=log,
                stderr=log,
                timeout=timeout_secs,
            )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Training exceeded {timeout_secs}s — treating as crash")
        return False
    except Exception as e:
        print(f"Training subprocess error: {e}")
        return False


def parse_run_log() -> dict:
    """Extract metrics from run.log. Returns {} if not found."""
    metrics = {}
    if not RUN_LOG.exists():
        return metrics
    for line in RUN_LOG.read_text().splitlines():
        for key in ("val_accuracy", "val_auc", "peak_vram_mb"):
            if line.startswith(f"{key}:"):
                try:
                    metrics[key] = float(line.split(":", 1)[1].strip())
                except (IndexError, ValueError):
                    pass
    return metrics


def tail_log(n: int = 25) -> str:
    if not RUN_LOG.exists():
        return "(no log)"
    lines = RUN_LOG.read_text().splitlines()
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# Scheduled idea I/O
# ---------------------------------------------------------------------------

def load_idea() -> dict | None:
    if SCHEDULED_IDEA_PATH.exists():
        with open(SCHEDULED_IDEA_PATH) as f:
            return json.load(f)
    return None


def clear_idea():
    if SCHEDULED_IDEA_PATH.exists():
        SCHEDULED_IDEA_PATH.unlink()


# ---------------------------------------------------------------------------
# Cron setup helper
# ---------------------------------------------------------------------------

def print_cron_setup():
    uv_path = "uv"
    try:
        uv_path = subprocess.check_output(["which", "uv"], text=True).strip()
    except Exception:
        pass

    cron_line = f"0 23 * * * cd {REPO_DIR} && {uv_path} run schedule_run.py >> {REPO_DIR}/cron.log 2>&1"
    print("─" * 60)
    print("Add this line to your crontab to run at 11pm every night:")
    print()
    print(f"  {cron_line}")
    print()
    print("To install it automatically:")
    print(f'  (crontab -l 2>/dev/null; echo "{cron_line}") | crontab -')
    print()
    print("To verify it was added:")
    print("  crontab -l")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(idea_text: str, chat_id=None):
    from notify_telegram import send

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Scheduled run: {idea_text}")
    print(f"{'='*60}\n")

    # Notify start
    send(f"🚀 <b>Starting overnight run</b>\n<i>{idea_text}</i>", chat_id=chat_id)

    # Parse and apply hyperparameters
    changes = parse_hyperparams(idea_text)
    applied = apply_hyperparams(changes) if changes else []

    if applied:
        commit_hash = git_commit_changes(f"sched: {', '.join(applied)}")
    else:
        print("No hyperparameter changes to apply — running current train_sentiment.py")
        commit_hash = ""

    # Run training
    success = run_training()
    elapsed = time.time() - t_start
    metrics = parse_run_log()

    if not success or not metrics:
        # Revert any patches and notify failure
        if applied:
            revert_train_script()
        error_tail = tail_log(20)
        msg = (
            f"💥 <b>CRASH</b> after {elapsed/60:.0f}min\n"
            f"<i>{idea_text}</i>\n\n"
            f"<code>{error_tail[-800:]}</code>"
        )
        print(msg)
        send(msg, chat_id=chat_id)
        clear_idea()
        return

    acc = metrics.get("val_accuracy", 0.0)
    auc = metrics.get("val_auc", 0.0)
    vram_mb = metrics.get("peak_vram_mb", 0.0)
    vram_gb = vram_mb / 1024

    # Decide keep/discard by checking if accuracy beats the last recorded best
    # (simple heuristic — agent loop in program_sentiment.md does the full comparison)
    result_icon = "✅" if acc >= 0.75 else "❌"
    applied_str = f"\nApplied: <code>{', '.join(applied)}</code>" if applied else ""
    hash_str = f"  <code>{commit_hash}</code>" if commit_hash else ""

    msg = (
        f"{result_icon} <b>Run complete</b>{hash_str} ({elapsed/60:.0f}min)\n\n"
        f"<b>val_accuracy: {acc:.4f}</b>\n"
        f"val_auc: {auc:.4f}\n"
        f"memory: {vram_gb:.1f} GB\n"
        f"{applied_str}\n\n"
        f"<i>{idea_text}</i>"
    )
    print(msg)
    send(msg, chat_id=chat_id)
    clear_idea()


def main():
    parser = argparse.ArgumentParser(description="Run scheduled overnight experiment")
    parser.add_argument("--setup", action="store_true", help="Print crontab entry and exit")
    parser.add_argument(
        "--now", type=str, metavar="IDEA",
        help="Run immediately with this idea text (skips reading scheduled_idea.json)",
    )
    args = parser.parse_args()

    if args.setup:
        print_cron_setup()
        return

    if args.now:
        run(idea_text=args.now, chat_id=None)
        return

    # Normal cron path: read idea from disk
    idea_data = load_idea()
    if not idea_data:
        print("No scheduled idea found.")
        print("Schedule one via Telegram (/idea command) or run:")
        print("  uv run schedule_run.py --now 'DEPTH=6 LM_LOSS_WEIGHT=0.1'")
        sys.exit(0)

    run(idea_text=idea_data["idea"], chat_id=idea_data.get("chat_id"))


if __name__ == "__main__":
    main()
