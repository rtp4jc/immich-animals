#!/usr/bin/env python
"""Parse sources into manifests and eyeball them on one contact sheet.

uv run python scripts/data.py parse dogfacenet
uv run python scripts/data.py inspect dogfacenet --rows 6
"""

import argparse
import logging
from pathlib import Path

from animal_id.common.logging_config import setup_logging
from animal_id.data import sources
from animal_id.data.sample import Source
from animal_id.data.visualize import contact_sheet, summary

logger = setup_logging(__name__, logging.INFO)

OUT_DIR = Path("outputs/data_inspect")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    parse = sub.add_parser("parse", help="(Re)write each source's manifest.")
    parse.add_argument("names", nargs="+", type=Source, choices=sources.SOURCES)
    inspect = sub.add_parser("inspect", help="Summary table and contact sheet.")
    inspect.add_argument("names", nargs="+", type=Source, choices=Source)
    inspect.add_argument("--rows", type=int, default=4, help="Rows per source.")
    inspect.add_argument("--cols", type=int, default=6)
    inspect.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.command == "parse":
        for name in args.names:
            sources.parse(name)
        return

    samples = [s for name in args.names for s in sources.load(name)]
    rows = summary(samples)
    table = [rows[0].keys()] + [row.values() for row in rows]
    logger.info("\n" + "\n".join("  ".join(f"{v:>10}" for v in r) for r in table))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{'+'.join(args.names)}.png"
    contact_sheet(samples, args.rows, args.cols, args.seed).save(path)
    logger.info(f"Contact sheet: {path}")


if __name__ == "__main__":
    main()
