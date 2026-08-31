"""Build the exploratory-pass report as a standalone HTML document.

Two outputs from the same body:

  --artifact  the fragment the Artifact publisher expects, which supplies its
              own <!doctype>, <head> and <body> wrapper;
  --standalone  a complete, valid HTML document that opens in any browser,
              prints to PDF, and can be emailed or committed as-is.

Both are self-contained: figures are embedded as data URIs and the only
external request is the Google Fonts stylesheet, which degrades to the
declared fallback stack when offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Exploratory pass over 44 sessions of the ODyn ket/xyl 16-odor dataset.">
"""
TAIL = """
</body>
</html>
"""


def standalone(body: str) -> str:
    """Wrap the artifact fragment in a complete HTML document."""
    marker = "</style>"
    cut = body.index(marker) + len(marker)
    return HEAD + body[:cut] + "\n</head>\n<body>\n" + body[cut:] + TAIL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, required=True,
                        help="the assembled report fragment")
    parser.add_argument("--standalone", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args(argv)
    body = args.body.read_text()
    if args.artifact:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(body)
        print("artifact fragment ->", args.artifact)
    if args.standalone:
        args.standalone.parent.mkdir(parents=True, exist_ok=True)
        document = standalone(body)
        args.standalone.write_text(document)
        print(f"standalone document -> {args.standalone} "
              f"({len(document)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
