"""Command line: build a library, check topics, search, or start the app.

    python -m collectivelens build --input pdfs/ --metadata metadata.csv
    python -m collectivelens eval
    python -m collectivelens search "youth participation"
    python -m collectivelens app
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config import USER_LIBRARY_DIR, Settings
from .evaluation import evaluate_topics
from .ingest import ingest_folder
from .library import build_documents, save_documents
from .llm import make_client
from .runtime import load_runtime
from .search import search
from .summarise import Summariser, load_provided_summaries


def _build(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    raw = ingest_folder(args.input, args.metadata)
    if not raw:
        print(f"No readable documents found in {args.input}", file=sys.stderr)
        return 1
    client = None if args.no_claude else make_client()
    if client is None and not args.no_claude and not args.summaries_file:
        print("No ANTHROPIC_API_KEY found: documents will be added without summaries. "
              "Search and topics still work.", file=sys.stderr)
    out = Path(args.out)
    summariser = Summariser(
        client, settings.claude_model, cache_path=out / "summary-cache.json",
        provided=load_provided_summaries(args.summaries_file),
        max_chars=settings.summary_max_chars, max_tokens=settings.summary_max_tokens)
    docs = build_documents(raw, summariser)
    save_documents(docs, out, is_sample=args.sample, name=args.name or out.name)
    problems = [(r.filename, p) for r in raw for p in r.problems]
    print(f"Saved {len(docs)} documents to {out}")
    for name, problem in problems:
        print(f"  note: {name}: {problem}", file=sys.stderr)
    return 0


def _eval(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.library:
        settings = replace(settings, library_dir=Path(args.library))
    runtime = load_runtime(settings)
    report = evaluate_topics(runtime.library)
    if report is None:
        print("No document has a reference topic (add a gold_topic column to the metadata file).")
        return 1
    print(report.to_markdown())
    return 0


def _search(args: argparse.Namespace) -> int:
    runtime = load_runtime(Settings.from_env())
    if runtime.warning:
        print(f"[{runtime.warning}]", file=sys.stderr)
    for hit in search(runtime.library, args.query, limit=args.limit):
        print(f"{hit.relative:5.0%}  {hit.doc.title}  ({hit.doc.org}, {hit.doc.year})  [{hit.topic}]")
    return 0


def _app(args: argparse.Namespace) -> int:
    from .ui import launch

    launch(share=args.share or None, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collectivelens", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="read PDFs/text files and save a library")
    b.add_argument("--input", required=True, help="folder with PDF, txt or md files")
    b.add_argument("--metadata", help="CSV with file_name, title, member_org, publication_year, source_url, gold_topic")
    b.add_argument("--out", default=str(USER_LIBRARY_DIR), help="where to save the library (default: ./library)")
    b.add_argument("--summaries-file", help="JSON with ready-made summaries (skips Claude for those files)")
    b.add_argument("--no-claude", action="store_true", help="do not call Claude even if a key is set")
    b.add_argument("--name", help="a name for the library, shown in the app")
    b.add_argument("--sample", action="store_true", help="mark the library as the bundled sample")
    b.set_defaults(run=_build)

    e = sub.add_parser("eval", help="compare automatic topics with gold_topic")
    e.add_argument("--library", help="library folder (default: your library, else the sample)")
    e.set_defaults(run=_eval)

    s = sub.add_parser("search", help="search from the terminal")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(run=_search)

    a = sub.add_parser("app", help="start the web app")
    a.add_argument("--share", action="store_true", help="create a temporary public link (off by default)")
    a.add_argument("--port", type=int, default=7860)
    a.set_defaults(run=_app)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
