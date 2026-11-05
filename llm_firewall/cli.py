"""Command-line scanner.

    llm-firewall scan "Ignore previous instructions"        # input-side verdict
    echo "my email is a@b.com" | llm-firewall scan-output    # output-side redaction
    llm-firewall scan --file prompts.txt --jsonl             # one verdict per line
    llm-firewall serve --port 8000                           # run the proxy

(or `python -m llm_firewall.cli ...` without installing the package)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable

from llm_firewall.config import Settings
from llm_firewall.pipeline import Firewall

_COLORS = {"allow": "\033[32m", "log": "\033[33m", "redact": "\033[33m", "block": "\033[31m"}
_RESET = "\033[0m"


def _inputs(args: argparse.Namespace) -> Iterable[str]:
    if args.text:
        yield " ".join(args.text)
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line.rstrip("\n")
    else:
        data = sys.stdin.read()
        if args.jsonl:
            yield from (ln for ln in data.splitlines() if ln.strip())
        else:
            yield data


def _paint(action: str, enable: bool) -> str:
    return f"{_COLORS.get(action, '')}{action.upper()}{_RESET}" if enable else action.upper()


def cmd_scan(args: argparse.Namespace) -> int:
    fw = Firewall(Settings.from_env(), load_transformer=not args.no_transformer)
    color = sys.stdout.isatty() and not args.json and not args.jsonl
    worst = 0
    for text in _inputs(args):
        v = fw.scan_input(text)
        worst = max(worst, {"allow": 0, "log": 1, "block": 2}.get(v.action.value, 0))
        if args.json or args.jsonl:
            out = v.to_dict()
            out["text"] = text if args.jsonl else None
            print(json.dumps(out, ensure_ascii=False) if args.jsonl else json.dumps(out, indent=2, ensure_ascii=False))
            continue
        print(f"verdict: {_paint(v.action.value, color)}   risk={v.risk:.3f}")
        for layer, score in v.scores.items():
            print(f"  {layer:<12} {'n/a' if score is None else f'{score:.3f}'}")
        for h in v.rule_hits:
            print(f"  rule  {h['rule']} ({h['category']}, {h['severity']}): {h['matched']!r}")
        if v.obfuscation_signals:
            print(f"  obfuscation: {', '.join(v.obfuscation_signals)}")
        for d in v.decoded_payloads:
            print(f"  decoded: {d[:120]!r}")
        print(f"  latency: {v.latency_ms['total']:.2f} ms")
    # Exit code lets shell scripts / CI gate on the verdict: 0 allow, 1 log, 2 block.
    return worst if args.exit_code else 0


def cmd_scan_output(args: argparse.Namespace) -> int:
    fw = Firewall(Settings.from_env(), load_transformer=False)
    for text in _inputs(args):
        v = fw.scan_output(text, canary=args.canary or "")
        if args.json or args.jsonl:
            print(json.dumps(v.to_dict(), ensure_ascii=False))
        else:
            print(f"verdict: {v.action.value.upper()}  findings={v.findings}")
            print(v.text)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("llm_firewall.proxy.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llm-firewall", description="Scan text with LLM-Firewall.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("text", nargs="*", help="text to scan (default: read stdin)")
        sp.add_argument("--file", help="scan each non-empty line of a file")
        sp.add_argument("--json", action="store_true", help="print full JSON verdict")
        sp.add_argument("--jsonl", action="store_true", help="one JSON object per line")

    s = sub.add_parser("scan", help="input-side: prompt injection / jailbreak detection")
    common(s)
    s.add_argument("--no-transformer", action="store_true", help="skip Layer C even if a model is present")
    s.add_argument("--exit-code", action="store_true", help="exit 0/1/2 for allow/log/block")
    s.set_defaults(func=cmd_scan)

    o = sub.add_parser("scan-output", help="output-side: PII / secret / canary detection and redaction")
    common(o)
    o.add_argument("--canary", help="canary token that must not appear in the text")
    o.set_defaults(func=cmd_scan_output)

    v = sub.add_parser("serve", help="run the proxy with uvicorn")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--reload", action="store_true")
    v.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
