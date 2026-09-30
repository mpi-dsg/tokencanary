from __future__ import annotations

import argparse
import json
import logging
import sys

from .auditor import Auditor
from .calibration import Calibration
from .config import load_config


def proxy(args) -> None:
    from .proxy import make_server

    auditor = Auditor(load_config(args.config))
    server = make_server(args.upstream, auditor, args.host, args.port, inject_logprobs=not args.no_inject)
    print(f"export OPENAI_BASE_URL=http://{args.host}:{args.port}/v1   (-> {args.upstream})", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n" + auditor.report(), file=sys.stderr)


def report(args) -> None:
    from .report import load_log, load_state, summarize

    config = load_config(args.config)
    print(summarize(load_log(config.log), config, load_state(config.state)))


def calibrate(args) -> None:
    from .commission import calibrate as run
    from .scoring import FIREWORKS, EchoScorer
    from .tokenizer import TokenizerRegistry

    config = load_config(args.config)
    tok = TokenizerRegistry(config.tokenizers, config.guess_hf).get(args.served_model)
    if tok is None:
        sys.exit(f"no tokenizer for {args.served_model}; add it under [tokenizers] in the config")
    with open(args.prompts) as f:
        lines = [line for line in f if line.strip()]
    prompts = [json.loads(ln)["messages"] if ln.startswith("{") else [{"role": "user", "content": ln.strip()}] for ln in lines]
    if args.system:
        prompts = [[{"role": "system", "content": args.system}, *p] for p in prompts]
    scorer = EchoScorer(args.scorer_model, base_url=args.base_url or FIREWORKS)
    cal = Calibration(args.out)
    kept = run(scorer, args.served_model, tok, prompts, cal, args.temperature, args.top_p, args.max_tokens, args.language)
    cal.save()
    print(f"kept {kept} of {len(prompts)} responses; noise margin {cal.margins.get(args.served_model, 0.0):.2f} nats")
    for key, scores in cal.scores.items():
        print(f"{key}: n={len(scores)}, non-canonical={sum(s != 0 for s in scores)}")


def _timestamp(iso: str | None) -> float | None:
    from datetime import datetime

    return datetime.fromisoformat(iso).timestamp() if iso else None


def calibrate_counts(args) -> None:
    from .commission import count_gaps
    from .report import load_log

    config = load_config(args.config)
    cal = Calibration(args.out)
    added = count_gaps(load_log(args.log or config.log), args.model, cal, args.host, _timestamp(args.since), _timestamp(args.until))
    cal.save()
    print(f"added {added} gaps; per key: " + ", ".join(f"{k}: {len(v)}" for k, v in cal.gaps.items()))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="tokencanary")
    sub = p.add_subparsers(dest="cmd", required=True)
    config_help = "TOML config (default ~/.tokencanary/config.toml)"

    sp = sub.add_parser("proxy", help="auditing reverse proxy on localhost")
    sp.add_argument("--upstream", required=True, help="e.g. https://api.openai.com")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8787)
    sp.add_argument("--no-inject", action="store_true", help="do not add logprobs to requests")
    sp.add_argument("--config", help=config_help)
    sp.set_defaults(func=proxy)

    sp = sub.add_parser("report", help="summarize the audit log")
    sp.add_argument("--config", help=config_help)
    sp.set_defaults(func=report)

    sp = sub.add_parser("calibrate", help="collect honest calibration data from the scoring provider")
    sp.add_argument("--served-model", required=True, help="model name used in the audited API requests")
    sp.add_argument("--scorer-model", required=True, help="the same weights on the scoring provider")
    sp.add_argument("--base-url", help="scoring provider API (default Fireworks)")
    sp.add_argument("--prompts", required=True, help="one prompt per line, or JSONL with `messages`")
    sp.add_argument("--system", help="system prompt your application sends")
    sp.add_argument("--language", help="calibration language tag (default: dominant script of each response)")
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--top-p", type=float, default=1.0)
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.add_argument("--out", required=True, help="calibration JSON to create or extend")
    sp.add_argument("--config", help=config_help)
    sp.set_defaults(func=calibrate)

    sp = sub.add_parser("calibrate-counts", help="calibrate the count-only rules on a trusted period of an endpoint")
    sp.add_argument("--model", required=True, help="model name used in the audited API requests")
    sp.add_argument("--host", help="audited host (default: all hosts in the log)")
    sp.add_argument("--since", help="start of the trusted period, ISO time")
    sp.add_argument("--until", help="end of the trusted period, ISO time")
    sp.add_argument("--log", help="audit log (default: from the config)")
    sp.add_argument("--out", required=True, help="calibration JSON to create or extend")
    sp.add_argument("--config", help=config_help)
    sp.set_defaults(func=calibrate_counts)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args.func(args)
