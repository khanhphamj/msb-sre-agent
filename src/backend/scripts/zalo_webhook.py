"""Zalo Bot admin helper. It reads the bot token and the webhook secret from Access Control (the same providers the
agent uses), so nothing secret is typed or printed.

  uv run python -m scripts.zalo_webhook me                                          # check the bot token
  uv run python -m scripts.zalo_webhook set --url https://<runtime-endpoint>/webhook/zalo

Locally it needs .greennode.json with "agent_identity" (see /agentbase-build-identity). The runtime image does not
ship this script's credentials (.dockerignore).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from app.channels.zalo import WEBHOOK_PATH, SecretCache, ZaloClient  # noqa: E402
from app.config import get_settings  # noqa: E402


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("me", help="check the bot token (prints the bot name)")
    setter = sub.add_parser("set", help="point the bot's webhook at this URL")
    setter.add_argument("--url", required=True, help=f"https://<runtime-endpoint>{WEBHOOK_PATH}")
    args = parser.parse_args(argv)

    settings = get_settings()
    cache = SecretCache(settings.zalo_secret_ttl_s)
    client = ZaloClient(settings, cache)
    try:
        if args.cmd == "me":
            answer = await client.call("getMe", get=True)
            if not answer.get("ok"):
                print(f"FAIL: {answer.get('error') or answer.get('http') or 'rejected'}")
                return 1
            info = answer.get("result") if isinstance(answer.get("result"), dict) else {}
            print(f"OK: bot '{info.get('display_name') or info.get('account_name') or '?'}'")
            return 0
        if not args.url.startswith("https://") or not args.url.endswith(WEBHOOK_PATH):
            print(f"FAIL: --url must be https://... and end with {WEBHOOK_PATH}")
            return 2
        secret = await cache.get(settings.zalo_secret_provider)
        if not secret:
            print(
                f"FAIL: cannot read provider '{settings.zalo_secret_provider}' from Access Control"
            )
            return 1
        answer = await client.call("setWebhook", {"url": args.url, "secret_token": secret})
        if not answer.get("ok"):
            print(
                f"FAIL: {answer.get('error') or answer.get('http') or answer.get('description') or 'rejected'}"
            )
            return 1
        print(f"OK: webhook set to {args.url}")
        return 0
    finally:
        await client.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
