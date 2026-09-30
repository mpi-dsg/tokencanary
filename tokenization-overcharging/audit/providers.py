"""Thin OpenAI-compatible client shared across the four resale providers."""
import os
import time

import requests


class ProviderClient:
    def __init__(self, name, base_url, api_key_env, max_tokens_field="max_tokens",
                 want_logprobs=False):
        self.name = name
        # Newer OpenAI models reject the legacy `max_tokens` and require
        # `max_completion_tokens`; the resale providers all still use the
        # legacy name. Parameterized rather than special-cased at call sites.
        self.max_tokens_field = max_tokens_field
        # OpenAI returns one logprobs entry per generated token, which gives a
        # billed-count cross-check that needs no tokenizer at all.
        self.want_logprobs = want_logprobs
        self.base_url = base_url.rstrip("/")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} not set — is .env populated and loaded?"
            )
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self):
        """Zero-cost catalog lookup. Returns a list of model id strings."""
        resp = requests.get(
            f"{self.base_url}/models", headers=self._headers, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        # Most providers use the OpenAI-compatible {"data": [...]} envelope,
        # but Together's /models returns a raw JSON array.
        items = data.get("data", []) if isinstance(data, dict) else data
        return [m["id"] for m in items]

    def chat_completion(
        self, model, system_prompt, user_prompt, temperature, top_p, max_tokens,
        retries=3,
    ):
        # Omit the system turn entirely when there's no system prompt. Sending
        # {"role": "system", "content": null} is rejected outright by some
        # providers (Together 400s, DeepInfra 422s) rather than ignored, which
        # reads as "model unavailable" and is easy to misdiagnose. The main
        # study always passed a real string (GENERATION["system_prompt"]), so
        # no collected data is affected -- this only bit ad-hoc probes.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            self.max_tokens_field: max_tokens,
        }
        if self.want_logprobs:
            payload["logprobs"] = True
        last_err = None
        for attempt in range(retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                    # Some models (e.g. Together's gemma-4-31B-it) appear to
                    # spend a long time on hidden generation before any
                    # visible content streams out -- observed >150s on a
                    # 300-max_tokens request. Generous timeout to avoid
                    # spurious failures rather than genuine unavailability.
                    timeout=240,
                )
                if resp.status_code in (402, 429):
                    # Not every 429 is rate limiting. Z.ai returns 429 with
                    # code 1113 ("Insufficient balance or no resource
                    # package") for an exhausted account, and OpenRouter
                    # returns 402 for the same condition. Backing off and
                    # retrying those is pointless: it turns a hard stop into
                    # a slow, silent churn through the whole prompt list that
                    # looks exactly like a hung process. Fail fast instead so
                    # the real cause is visible immediately.
                    body = resp.text[:300]
                    if resp.status_code == 402 or any(
                        s in body.lower()
                        for s in ("insufficient balance", "recharge",
                                  "quota", "billing", "payment")
                    ):
                        raise RuntimeError(
                            f"{self.name}: account/billing error, not retryable: "
                            f"HTTP {resp.status_code} {body}"
                        )
                    time.sleep(2 ** attempt * 2)
                    continue
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                text = message.get("content") or ""
                # Reasoning-capable models return hidden chain-of-thought in
                # a field separate from `content`, and bill completion_tokens
                # for both combined. Field name varies by provider/model --
                # Together's gemma-4-31B-it uses `reasoning`, Fireworks'
                # gpt-oss uses `reasoning_content`. Missing this makes a
                # faithful provider look like it's massively overcharging.
                # Concatenate in generation order (reasoning precedes the
                # visible answer) so canonical recomputation covers
                # everything the provider actually billed for.
                reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                usage = data.get("usage", {})
                lp = (data["choices"][0].get("logprobs") or {}).get("content")
                return {
                    "text": text,
                    "reasoning_text": reasoning,
                    # One entry per generated token when available: an
                    # internal-consistency check on the billed count that
                    # requires no tokenizer and no trust in one.
                    "logprob_entries": len(lp) if lp else None,
                    "billed_completion_tokens": usage.get("completion_tokens"),
                    "raw_usage": usage,
                    "response_id": data.get("id"),
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "raw": data,
                }
            except Exception as e:  # noqa: BLE001 - want broad retry on transient errors
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{self.name}: request failed after {retries} retries: {last_err}")

    def generation_stats(self, generation_id, retries=3):
        """OpenRouter-only: fetch post-hoc stats including native provider
        token counts and which backend actually served the request."""
        if self.name != "openrouter":
            return None
        for attempt in range(retries):
            resp = requests.get(
                f"{self.base_url}/generation",
                headers=self._headers,
                params={"id": generation_id},
                timeout=30,
            )
            if resp.status_code == 404:
                # Stats can take a moment to become available after the response.
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json().get("data", {})
        return None


def build_client(provider_key):
    from config import PROVIDERS

    cfg = PROVIDERS[provider_key]
    return ProviderClient(
        provider_key, cfg["base_url"], cfg["api_key_env"],
        max_tokens_field=cfg.get("max_tokens_field", "max_tokens"),
        want_logprobs=cfg.get("want_logprobs", False),
    )
