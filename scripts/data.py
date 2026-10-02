#!/usr/bin/env python
"""Fetch, prepare and parse sources into manifests; eyeball them on a contact sheet.

uv run python scripts/data.py parse dogfacenet
uv run python scripts/data.py inspect dogfacenet --rows 6
uv run python scripts/data.py fetch open_images --max-per-class 13000
"""

import argparse
import logging
from pathlib import Path

from animal_id.common.constants import DATA_DIR
from animal_id.common.logging_config import setup_logging
from animal_id.data import sources
from animal_id.data.sample import Source
from animal_id.data.sources import cat_individuals, open_images
from animal_id.data.visualize import contact_sheet, summary

logger = setup_logging(__name__, logging.INFO)

OUT_DIR = Path("outputs/data_inspect")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download a source's images.")
    fetch.add_argument("name", type=Source, choices=[Source.OPEN_IMAGES])
    fetch.add_argument("--max-per-class", type=int, required=True)
    fetch.add_argument("--workers", type=int, default=32)
    prepare = sub.add_parser(
        "prepare", help="Resize and box a manually downloaded source."
    )
    prepare.add_argument("name", type=Source, choices=[Source.CAT_INDIVIDUALS])
    parse = sub.add_parser("parse", help="(Re)write each source's manifest.")
    parse.add_argument("names", nargs="+", type=Source, choices=sources.SOURCES)
    inspect = sub.add_parser("inspect", help="Summary table and contact sheet.")
    inspect.add_argument("names", nargs="+", type=Source, choices=Source)
    inspect.add_argument("--rows", type=int, default=4, help="Rows per source.")
    inspect.add_argument("--cols", type=int, default=6)
    inspect.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.command == "fetch":
        open_images.fetch(DATA_DIR, args.max_per_class, args.workers)
        return
    if args.command == "prepare":
        cat_individuals.prepare(DATA_DIR)
        return
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
