"""
Telegram scoring bot for financial sentiment.

Run this alongside (or after) the overnight training loop.
Send any text to the bot and it replies with BULLISH/BEARISH + confidence.

Long transcripts (paste an entire earnings call Q&A section) work fine —
the bot chunks them automatically.

Usage:
    uv run telegram_bot.py

Environment variables required:
    TELEGRAM_BOT_TOKEN  — from @BotFather
    TELEGRAM_CHAT_ID    — (optional) restrict to one user/chat for security

Setup:
    See notify_telegram.py for step-by-step setup instructions.

Special commands the bot understands:
    /start   — welcome message
    /help    — show usage
    /status  — show loaded model info
    <any text> — score it for bullish/bearish sentiment

The bot uses long-polling (no webhook needed, works behind any firewall).
"""

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

# ---------------------------------------------------------------------------
# Telegram API helpers (raw HTTP — no extra dependencies)
# ---------------------------------------------------------------------------

def api_call(method: str, **kwargs) -> dict:
    """Call a Telegram Bot API method."""
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
# Model loading (lazy — loaded on first message)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_device = None
_model_info = "not loaded"


def get_model():
    global _model, _tokenizer, _device, _model_info
    if _model is not None:
        return _model, _tokenizer, _device

    from prepare_sentiment import SentimentTokenizer, SENTIMENT_CACHE_DIR
    from score_transcript import load_model

    model_path = SENTIMENT_CACHE_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}.\n"
            "Run training first: uv run train_sentiment.py"
        )

    _tokenizer = SentimentTokenizer.from_cache()
    _model, _device = load_model(model_path)
    _model.eval()

    import torch
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
    """Score text and format a Telegram reply."""
    from score_transcript import score_text, chunk_tokens

    model, tokenizer, device = get_model()
    prob = score_text(text, model, tokenizer, device)

    label = "🟢 BULLISH" if prob > 0.5 else "🔴 BEARISH"
    confidence = prob if prob > 0.5 else 1.0 - prob
    bar_filled = round(confidence * 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)

    token_ids = tokenizer.encode(text)
    chunks = chunk_tokens(token_ids)

    reply = (
        f"{label}\n"
        f"<code>{bar}</code> {confidence:.1%}\n"
        f"Raw score: {prob:.4f}\n"
    )

    if len(chunks) > 1:
        reply += f"\n<i>{len(chunks)} chunks analysed (long transcript)</i>\n"
        lines = []
        import torch
        for i, chunk in enumerate(chunks):
            t = torch.tensor([chunk], dtype=torch.long, device=device)
            with torch.no_grad():
                logit = model(t)
            p = torch.sigmoid(logit).item()
            arrow = "↑" if p > 0.5 else "↓"
            lines.append(f"  chunk {i+1}: {p:.3f} {arrow}")
        reply += "\n".join(lines)

    return reply


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

WELCOME = (
    "👋 <b>Financial Sentiment Bot</b>\n\n"
    "Send me any financial text — earnings call excerpts, management guidance, "
    "analyst commentary — and I'll tell you whether it reads as "
    "<b>BULLISH</b> or <b>BEARISH</b>.\n\n"
    "Long transcripts are fine. Paste as much as you like.\n\n"
    "Commands: /help /status"
)

HELP = (
    "<b>How to use:</b>\n"
    "• Send any financial text\n"
    "• For a full transcript, paste it directly or send as a file\n\n"
    "<b>The model was trained on FinancialPhraseBank</b> — "
    "labeled financial news sentences. It generalises well to earnings "
    "call language but was not trained on full-length transcripts "
    "(that's what the overnight runs are for)."
)


def handle_message(msg: dict):
    chat_id = msg["chat"]["id"]
    text = msg.get("text", "").strip()
    # Document (text file) support
    if not text and "document" in msg:
        send_message(chat_id, "📎 File uploads aren't supported yet — paste the text directly.")
        return
    if not text:
        return

    # Access control
    if ALLOWED_CHAT_ID and str(chat_id) != str(ALLOWED_CHAT_ID):
        send_message(chat_id, "⛔ Unauthorised.")
        return

    # Commands
    if text.startswith("/start"):
        send_message(chat_id, WELCOME)
        return
    if text.startswith("/help"):
        send_message(chat_id, HELP)
        return
    if text.startswith("/status"):
        try:
            get_model()  # force load
            send_message(chat_id, f"✅ Model loaded\n{_model_info}")
        except Exception as e:
            send_message(chat_id, f"❌ Model not ready: {e}")
        return
    if text.startswith("/"):
        send_message(chat_id, "Unknown command. Try /help")
        return

    # Score the text
    send_typing(chat_id)
    try:
        reply = score_and_reply(text)
        send_message(chat_id, reply)
    except FileNotFoundError as e:
        send_message(chat_id, f"❌ {e}")
    except Exception as e:
        traceback.print_exc()
        send_message(chat_id, f"❌ Error: {e}")


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------

def main():
    print(f"Starting bot (long-polling)...")
    print(f"Chat filter: {ALLOWED_CHAT_ID or 'none (open to all)'}")

    # Announce readiness — eagerly load model
    try:
        get_model()
        print("Model pre-loaded.")
    except FileNotFoundError:
        print("Model not found — will retry on first message.")

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
            print(f"Poll error (attempt {consecutive_errors}): {e} — retrying in {wait}s")
            time.sleep(wait)


if __name__ == "__main__":
    main()
