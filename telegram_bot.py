"""
Telegram bot for financial sentiment scoring and overnight run scheduling.

Two modes in one bot:
  1. SCORING — send any financial text, get BULLISH/BEARISH back instantly.
  2. SCHEDULING — describe tonight's experiment idea; the bot confirms,
     saves it, and schedule_run.py picks it up at 11pm via cron.

Usage:
    uv run telegram_bot.py

Environment variables required:
    TELEGRAM_BOT_TOKEN  — from @BotFather
    TELEGRAM_CHAT_ID    — (optional) restrict to one user/chat for security

Setup:
    See notify_telegram.py for step-by-step BotFather setup instructions.

Commands:
    /start        — welcome message
    /help         — show all commands
    /status       — model info + pending scheduled run
    /idea <text>  — schedule tonight's experiment (asks for confirmation)
    /tonight <t>  — alias for /idea
    /cancel       — cancel pending scheduled run
    <any text>    — score it for bullish/bearish sentiment

The bot uses long-polling (no webhook needed, works behind any firewall).
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ALLOWED_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")  # leave empty to allow all

if not BOT_TOKEN:
    print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
    print("See notify_telegram.py for setup instructions.")
    sys.exit(1)

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

REPO_DIR = Path(__file__).parent
from prepare_sentiment import SENTIMENT_CACHE_DIR
SCHEDULED_IDEA_PATH = SENTIMENT_CACHE_DIR / "scheduled_idea.json"

# ---------------------------------------------------------------------------
# Per-chat confirmation state
# { chat_id: idea_text } — set while waiting for yes/no
# ---------------------------------------------------------------------------

_pending_confirmations: dict = {}

# ---------------------------------------------------------------------------
# Telegram API helpers
# ---------------------------------------------------------------------------

def api_call(method: str, **kwargs) -> dict:
    resp = requests.post(f"{API_BASE}/{method}", json=kwargs, timeout=35)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data["result"]


def send_message(chat_id, text: str, parse_mode="HTML"):
    try:
        api_call("sendMessage", chat_id=chat_id, text=text, parse_mode=parse_mode)
    except Exception as e:
        print(f"send_message failed: {e}")


def send_typing(chat_id):
    try:
        api_call("sendChatAction", chat_id=chat_id, action="typing")
    except Exception:
        pass


def get_updates(offset: int = 0) -> list:
    try:
        return api_call("getUpdates", timeout=30, offset=offset)
    except Exception as e:
        print(f"getUpdates error: {e}")
        return []


# ---------------------------------------------------------------------------
# Scheduled idea storage
# ---------------------------------------------------------------------------

def save_idea(chat_id: int, idea_text: str):
    """Persist idea to disk so schedule_run.py can pick it up at 11pm."""
    SENTIMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCHEDULED_IDEA_PATH, "w") as f:
        json.dump({"idea": idea_text, "chat_id": chat_id}, f)


def load_idea() -> dict | None:
    if SCHEDULED_IDEA_PATH.exists():
        with open(SCHEDULED_IDEA_PATH) as f:
            return json.load(f)
    return None


def clear_idea():
    if SCHEDULED_IDEA_PATH.exists():
        SCHEDULED_IDEA_PATH.unlink()


# ---------------------------------------------------------------------------
# Model loading (lazy)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_device = None
_model_info = "not loaded"


def get_model():
    global _model, _tokenizer, _device, _model_info
    if _model is not None:
        return _model, _tokenizer, _device

    from prepare_sentiment import SentimentTokenizer
    from score_transcript import load_model
    import torch

    model_path = SENTIMENT_CACHE_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}.\n"
            "Run: uv run train_sentiment.py"
        )

    _tokenizer = SentimentTokenizer.from_cache()
    _model, _device = load_model(model_path)
    _model.eval()

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    acc = ckpt.get("val_accuracy", 0)
    auc = ckpt.get("val_auc", 0)
    _model_info = f"val_accuracy={acc:.4f}, val_auc={auc:.4f}"
    print(f"Model loaded: {_model_info}")
    return _model, _tokenizer, _device


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_and_reply(text: str) -> str:
    from score_transcript import score_text, chunk_tokens
    import torch

    model, tokenizer, device = get_model()
    prob = score_text(text, model, tokenizer, device)

    label = "🟢 BULLISH" if prob > 0.5 else "🔴 BEARISH"
    confidence = prob if prob > 0.5 else 1.0 - prob
    bar_filled = round(confidence * 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)

    reply = (
        f"{label}\n"
        f"<code>{bar}</code> {confidence:.1%}\n"
        f"Raw score: {prob:.4f}\n"
    )

    token_ids = tokenizer.encode(text)
    chunks = chunk_tokens(token_ids)
    if len(chunks) > 1:
        reply += f"\n<i>{len(chunks)} chunks analysed (long transcript)</i>\n"
        for i, chunk in enumerate(chunks):
            t = torch.tensor([chunk], dtype=torch.long, device=device)
            with torch.no_grad():
                logit = model(t)
            p = torch.sigmoid(logit).item()
            arrow = "↑" if p > 0.5 else "↓"
            reply += f"  chunk {i+1}: {p:.3f} {arrow}\n"

    return reply.strip()


# ---------------------------------------------------------------------------
# Idea scheduling helpers
# ---------------------------------------------------------------------------

def _preview_idea_changes(idea_text: str) -> str:
    """Return a human-readable summary of what hyperparams will be changed."""
    import re
    # Recognised hyperparameters in train_sentiment.py
    KNOWN = {
        "DEPTH", "ASPECT_RATIO", "HEAD_DIM", "CLS_DROPOUT",
        "LM_LOSS_WEIGHT", "DEVICE_BATCH_SIZE", "EMBEDDING_LR",
        "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY",
        "WARMUP_RATIO", "WARMDOWN_RATIO", "WINDOW_PATTERN",
    }
    found = []
    for m in re.finditer(r'(\w+)\s*=\s*(["\']?[\w.]+["\']?)', idea_text, re.IGNORECASE):
        name = m.group(1).upper()
        if name in KNOWN:
            found.append(f"  • {name} = {m.group(2)}")
    if found:
        return "I'll automatically apply:\n" + "\n".join(found)
    return "No hyperparameter assignments detected — I'll run with current <code>train_sentiment.py</code> unchanged and use your idea as the description."


def confirm_prompt(idea_text: str) -> str:
    preview = _preview_idea_changes(idea_text)
    return (
        f"Got it! Tonight's 11pm run will try:\n\n"
        f"<i>{idea_text}</i>\n\n"
        f"{preview}\n\n"
        f"Schedule for <b>11pm</b> tonight? Reply <b>yes</b> to confirm or <b>no</b> to cancel."
    )


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

WELCOME = (
    "👋 <b>Financial Sentiment Bot</b>\n\n"
    "I can do two things:\n\n"
    "1. <b>Score text</b> — send any earnings call excerpt or financial statement and I'll reply BULLISH or BEARISH.\n\n"
    "2. <b>Schedule a run</b> — tell me what architecture idea to try tonight and I'll kick it off at 11pm via cron, then message you with the results.\n\n"
    "Commands: /help /idea /status /cancel"
)

HELP = (
    "<b>Scoring</b>\n"
    "Just send any financial text. Long transcripts are chunked automatically.\n\n"
    "<b>Scheduling a run</b>\n"
    "<code>/idea try LM_LOSS_WEIGHT=0.1 with DEPTH=6</code>\n"
    "<code>/tonight ASPECT_RATIO=128 for a wider model</code>\n"
    "Known hyperparams (auto-applied): DEPTH, ASPECT_RATIO, HEAD_DIM, "
    "CLS_DROPOUT, LM_LOSS_WEIGHT, DEVICE_BATCH_SIZE, MATRIX_LR, EMBEDDING_LR, "
    "WEIGHT_DECAY, WARMUP_RATIO, WARMDOWN_RATIO, WINDOW_PATTERN\n\n"
    "<b>Other commands</b>\n"
    "/status — model info + pending scheduled run\n"
    "/cancel — remove the pending scheduled run"
)


def handle_message(msg: dict):
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()

    if not text and "document" in msg:
        send_message(chat_id, "📎 File uploads aren't supported — paste the text directly.")
        return
    if not text:
        return

    if ALLOWED_CHAT_ID and str(chat_id) != str(ALLOWED_CHAT_ID):
        send_message(chat_id, "⛔ Unauthorised.")
        return

    # ── Handle yes/no confirmation if we're waiting for one ──────────────────
    if chat_id in _pending_confirmations:
        lowered = text.lower().strip("!. ")
        if lowered in ("yes", "y", "yep", "confirm", "ok", "sure", "✅", "👍"):
            idea = _pending_confirmations.pop(chat_id)
            save_idea(chat_id, idea)
            send_message(
                chat_id,
                "✅ <b>Scheduled for 11pm.</b>\n\n"
                "I'll message you when the run finishes. "
                "Make sure <code>cron</code> is running with <code>schedule_run.py</code> — "
                "see <code>uv run schedule_run.py --setup</code> for instructions.",
            )
        elif lowered in ("no", "n", "nope", "cancel", "❌", "👎"):
            _pending_confirmations.pop(chat_id)
            send_message(chat_id, "❌ Cancelled. Send another /idea whenever you're ready.")
        else:
            send_message(chat_id, "Please reply <b>yes</b> or <b>no</b>.")
        return

    # ── Slash commands ────────────────────────────────────────────────────────
    if text.startswith("/start"):
        send_message(chat_id, WELCOME)
        return

    if text.startswith("/help"):
        send_message(chat_id, HELP)
        return

    if text.startswith("/cancel"):
        if load_idea():
            clear_idea()
            send_message(chat_id, "✅ Scheduled run cancelled.")
        elif chat_id in _pending_confirmations:
            _pending_confirmations.pop(chat_id)
            send_message(chat_id, "✅ Pending idea cancelled.")
        else:
            send_message(chat_id, "Nothing to cancel.")
        return

    if text.startswith("/status"):
        lines = []
        try:
            get_model()
            lines.append(f"🧠 Model: {_model_info}")
        except FileNotFoundError:
            lines.append("🧠 Model: not trained yet")
        idea_data = load_idea()
        if idea_data:
            lines.append(f"\n📅 Scheduled for 11pm:\n<i>{idea_data['idea']}</i>")
        else:
            lines.append("\n📅 No run scheduled (use /idea to schedule one)")
        send_message(chat_id, "\n".join(lines))
        return

    if text.startswith("/idea ") or text.startswith("/tonight "):
        idea = text.split(" ", 1)[1].strip() if " " in text else ""
        if not idea:
            send_message(chat_id, "Usage: /idea <description>\nExample: /idea LM_LOSS_WEIGHT=0.1 DEPTH=6")
            return
        # Check if there's already a scheduled run
        existing = load_idea()
        if existing:
            send_message(
                chat_id,
                f"⚠️ There's already a run scheduled:\n<i>{existing['idea']}</i>\n\n"
                "Use /cancel first, then /idea again.",
            )
            return
        _pending_confirmations[chat_id] = idea
        send_message(chat_id, confirm_prompt(idea))
        return

    if text.startswith("/"):
        send_message(chat_id, "Unknown command. Try /help")
        return

    # ── Default: score the text ───────────────────────────────────────────────
    send_typing(chat_id)
    try:
        reply = score_and_reply(text)
        send_message(chat_id, reply)
    except FileNotFoundError as e:
        send_message(
            chat_id,
            f"❌ {e}\n\nTrain first: <code>uv run train_sentiment.py</code>"
        )
    except Exception as e:
        traceback.print_exc()
        send_message(chat_id, f"❌ Error: {e}")


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def main():
    print("Starting bot (long-polling)...")
    print(f"Chat filter: {ALLOWED_CHAT_ID or 'none (open to all)'}")

    try:
        get_model()
        print("Model pre-loaded.")
    except FileNotFoundError:
        print("No model yet — will try again on first scoring request.")

    offset = 0
    consecutive_errors = 0

    while True:
        try:
            updates = get_updates(offset)
            consecutive_errors = 0
            for update in updates:
                offset = update["update_id"] + 1
                if "message" in update:
                    handle_message(update["message"])
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            consecutive_errors += 1
            wait = min(2 ** consecutive_errors, 60)
            print(f"Poll error ({consecutive_errors}): {e} — retrying in {wait}s")
            time.sleep(wait)


if __name__ == "__main__":
    main()
