from __future__ import annotations

import argparse
import json
import logging
import sys

from .auditor import Auditor
from .calibration import Calibration
from .config import load_config


def _pairs(values: list[str] | None) -> dict[str, str]:
    try:
        return dict(v.split("=", 1) for v in values or [])
    except ValueError:
        sys.exit("expected MODEL=VALUE")


def proxy(args) -> None:
    from .proxy import make_server
    from .scoring import LocalHFScorer

    auditor = Auditor(load_config(args.config), scorers={m: LocalHFScorer(hf) for m, hf in _pairs(args.scorer).items()})
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
    from .scoring import LocalHFScorer, commission

    with open(args.prompts) as f:
        lines = [line for line in f if line.strip()]
    prompts = [json.loads(ln)["messages"] if ln.startswith("{") else [{"role": "user", "content": ln.strip()}] for ln in lines]
    if args.system:
        prompts = [[{"role": "system", "content": args.system}, *p] for p in prompts]
    cal = Calibration(args.out)
    served = args.served_model or args.model_id
    commission(LocalHFScorer(args.model_id), served, prompts, cal, args.temperature, args.top_p, args.max_tokens, args.language)
    cal.save()
    for key, scores in cal.scores.items():
        print(f"{key}: n={len(scores)}, non-canonical={sum(s != 0 for s in scores)}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="tokencanary")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("proxy", help="auditing reverse proxy on localhost")
    sp.add_argument("--upstream", required=True, help="e.g. https://api.openai.com")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8787)
    sp.add_argument("--scorer", action="append", metavar="MODEL=HF_ID", help="local reference weights (likelihood test)")
    sp.add_argument("--no-inject", action="store_true", help="do not add logprobs to requests")
    sp.add_argument("--config", help="TOML config (default ~/.tokencanary/config.toml)")
    sp.set_defaults(func=proxy)

    sp = sub.add_parser("report", help="summarize the audit log")
    sp.add_argument("--config", help="TOML config (default ~/.tokencanary/config.toml)")
    sp.set_defaults(func=report)

    sp = sub.add_parser("calibrate", help="collect honest likelihood scores from local generations")
    sp.add_argument("--model-id", required=True, help="HF weights identical to the served model")
    sp.add_argument("--served-model", help="model name used in API requests")
    sp.add_argument("--prompts", required=True, help="one prompt per line, or JSONL with `messages`")
    sp.add_argument("--system", help="system prompt your application sends")
    sp.add_argument("--language", help="calibration language tag (default: dominant script of each response)")
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--top-p", type=float, default=1.0)
    sp.add_argument("--max-tokens", type=int, default=512)
    sp.add_argument("--out", required=True, help="calibration JSON to create or extend")
    sp.set_defaults(func=calibrate)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args.func(args)
