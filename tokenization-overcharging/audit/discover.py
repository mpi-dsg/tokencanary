"""Zero-cost pre-flight check: confirm every model_id in
config.PROVIDER_MODEL_MAP still exists in each provider's live catalog before
audit.py spends any money. Run this before every real audit run, not just
once -- reseller catalogs change (see implementation-notes.md deviations).
"""
import sys

import env

env.ensure_loaded()

from config import PROVIDER_MODEL_MAP  # noqa: E402
from providers import build_client  # noqa: E402


def main():
    all_ok = True
    for provider_key, model_entries in PROVIDER_MODEL_MAP.items():
        print(f"\n=== {provider_key} ===")
        try:
            client = build_client(provider_key)
        except RuntimeError as e:
            print(f"  SKIP (no key): {e}")
            all_ok = False
            continue

        try:
            live_ids = set(client.list_models())
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR listing models: {e}")
            all_ok = False
            continue

        for short_key, entry in model_entries.items():
            model_id = entry["model_id"]
            if model_id in live_ids:
                print(f"  OK   {short_key}: {model_id}")
            else:
                print(f"  FAIL {short_key}: {model_id} not found in live catalog "
                      f"({len(live_ids)} models listed) — reseller may have deprecated it")
                all_ok = False

    print()
    if all_ok:
        print("All configured models confirmed live. Safe to run audit.py.")
    else:
        print("One or more models missing or provider unreachable. "
              "Fix config.py before spending on audit.py.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
