"""
Send a training experiment result to Telegram.

Called by the agent loop after each experiment to push a progress update
to your phone without you needing to watch the terminal.

Usage:
    uv run notify_telegram.py "keep | acc=0.801 | auc=0.863 | LM_LOSS_WEIGHT=0.1"

Environment variables required:
    TELEGRAM_BOT_TOKEN  — from @BotFather on Telegram
    TELEGRAM_CHAT_ID    — your chat/user ID (see setup instructions below)

Setup:
    1. Open Telegram and search for @BotFather
    2. Send /newbot and follow the prompts → copy the token
    3. Start a conversation with your new bot (send it any message)
    4. Run:  curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
       Find "chat":{"id": <YOUR_CHAT_ID>} in the response
    5. Export:
          export TELEGRAM_BOT_TOKEN="123456789:AAH..."
          export TELEGRAM_CHAT_ID="987654321"
    6. Test:  uv run notify_telegram.py "hello from autoresearch"
"""

import os
import sys
import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


def send(text: str) -> bool:
    """Send a message. Returns True on success."""
    if not BOT_TOKEN or not CHAT_ID:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping notification")
        return False
    try:
        resp = requests.post(
            API_URL,
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Telegram notify failed: {e}")
        return False


def format_experiment_result(
    status: str,
    val_accuracy: float,
    val_auc: float,
    description: str,
    commit: str = "",
) -> str:
    icon = {"keep": "✅", "discard": "❌", "crash": "💥"}.get(status, "❓")
    lines = [
        f"{icon} <b>{status.upper()}</b>  {commit}",
        f"acc={val_accuracy:.4f}  auc={val_auc:.4f}",
        f"{description}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run notify_telegram.py '<message>'")
        sys.exit(1)
    message = " ".join(sys.argv[1:])
    ok = send(message)
    if ok:
        print("Sent.")
    else:
        sys.exit(1)
