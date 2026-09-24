# uv run python examples/smoke_test.py  (needs OPENAI_API_KEY and/or DEEPINFRA_API_KEY)

import logging
import os
import tempfile

from openai import OpenAI

import tokencanary

ENDPOINTS = [
    ("OPENAI_API_KEY", None, "gpt-4o-mini"),
    ("DEEPINFRA_API_KEY", "https://api.deepinfra.com/v1/openai", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
]
PROMPTS = [("en", "Name the oldest continuously inhabited city in one sentence."),
           ("de", "Nenne in einem Satz die älteste durchgehend bewohnte Stadt.")]


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    tmp = tempfile.mkdtemp()
    auditor = tokencanary.Auditor(log=os.path.join(tmp, "audit.jsonl"), state=os.path.join(tmp, "state.json"), background=False)
    for key, base_url, model in ENDPOINTS:
        if not os.environ.get(key):
            print(f"skipping {model}: {key} not set")
            continue
        client = OpenAI(api_key=os.environ[key], base_url=base_url, http_client=tokencanary.http_client(auditor))
        for language, prompt in PROMPTS:
            messages = [{"role": "user", "content": prompt}]
            headers = {"x-tokencanary-language": language}
            reply = client.chat.completions.create(model=model, messages=messages, max_tokens=100, extra_headers=headers)
            assert reply.choices[0].logprobs is None, "injected logprobs leaked to the caller"
            stream = client.chat.completions.create(model=model, messages=messages, max_tokens=100, stream=True, extra_headers=headers)
            text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
            print(f"{model} [{language}]: {text[:70]!r}")
    print("\n" + auditor.report())


if __name__ == "__main__":
    main()
