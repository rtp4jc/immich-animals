#!/usr/bin/env python
"""Build a held-out validation set for the Immich sidecar.

Two things the sidecar has to get right, and neither is measurable on the
training splits: does it find the dog in a photo, and does it *avoid* firing on
photos with no dog (every false positive becomes a junk "person" in Immich).

    data/sidecar-validation/
      identities/<slug>/*.jpg   photos of one individual dog
      negatives/*.jpg           photos with no dog at all
      manifest.json             every file, its label, source URL and licence

Sources: hand-picked Wikimedia Commons categories for named individual dogs
(free licences, but noisy — see NON_PHOTO), Commons "Quality images" categories
for negatives, and the Multi-pose Dog Dataset (CC BY 4.0, 30 MB) for a second
opinion on identity. MPDD images are reID-style crops, not snapshots, so the
manifest tags every file with its source and the evaluator scores them apart.

    uv run python scripts/fetch_validation_set.py
    uv run python scripts/fetch_validation_set.py --limit 3   # smoke run
"""

import argparse
import hashlib
import io
import json
import logging
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

from animal_id.common.constants import DATA_DIR
from animal_id.common.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

OUT_DIR = DATA_DIR / "sidecar-validation"
API = "https://commons.wikimedia.org/w/api.php"
# Commons blocks anonymous bots; a contactable UA is their stated requirement.
UA = "immich-animals-validation/1.0 (https://github.com/ryan/immich-animals)"
MIN_PHOTOS = 4  # fewer than this and an identity is useless for clustering

# Categories holding one named dog each, picked by hand from Category:Individual
# dogs / Category:Famous dogs for having >= MIN_PHOTOS real photographs. Groups
# ("Bo and Sunny together"), fictional dogs, mascot costumes and statue-only
# categories are out; so are the historic ones that are mostly contact sheets.
IDENTITY_CATEGORIES = {
    "messi-actor": "Category:Messi (dog)",
    "bo-obama": "Category:Bo (dog)",
    "sunny-obama": "Category:Sunny (dog)",
    "commander-biden": "Category:Commander (dog)",
    "champ-biden": "Category:Champ (dog)",
    "major-biden": "Category:Major (dog)",
    "bailey-warren": "Category:Bailey (Elizabeth Warren's dog)",
    "babydog-justice": "Category:Babydog",
    "conan-mwd": "Category:President Trump Welcomes Conan the Military Working Dog to the White House",
    "patron-ukraine": "Category:Patron (dog)",
    "taro-shiba": "Category:Taro the Shiba Inu",
    "kabosu-doge": "Category:Kabosu (dog)",
    "vorli": "Category:Vorli",
    "zazu": "Category:Zazu the dog",
    "andromeda-galaktika": "Category:Andromeda Galaktika",
    "eddi": "Category:Eddi (dog)",
    "skinna": "Category:Skinna (dog)",
    "madsen-esbjerg": "Category:Madsen the dog (Esbjerg)",
    "resistencia": "Category:Resistência (dog)",
    "sansao": "Category:Sansão (dog)",
    "buddy-clinton": "Category:Buddy (dog)",
    "koko": "Category:Koko (dog)",
    "wikiwuff": "Category:WikiWuff",
    "yume-akita": "Category:Yume",
    "mathilda": "Category:Mathilda (dog)",
    "sully-bush": "Category:Sully (dog)",
    "victory": "Category:Victory (dog)",
    "millie-bush": "Category:Millie (dog)",
    "millie-reagan": "Category:Millie (dog of Ronald Reagan)",
    "spot-fetcher-bush": "Category:Spot Fetcher",
    "bobi": "Category:Bobi",
    "mr-newman-pug": "Category:Mr Newman (pug)",
    "pecas": "Category:Pecas",
    "ddewey": "Category:DDewey",
    "eli-mwd": "Category:Eli (military working dog)",
    "duffy": "Category:Duffy (dog)",
    "whoopie": "Category:Whoopie (dog)",
    "lola": "Category:Lola (dog)",
    "medor": "Category:Médor (dog)",
    "gem-guide-dog": "Category:Gem (guide dog)",
    "gino": "Category:Gino (dog)",
    "loukanikos": "Category:Loukanikos",
    "freebo": "Category:Freebo (dog)",
    "monk": "Category:Monk (dog)",
    "charlie": "Category:Charlie (dog)",
    "brownie": "Category:Brownie (dog)",
    "willie": "Category:Willie (dog)",
    "tip": "Category:Tip (dog)",
    "rex": "Category:Rex (dog)",
    "teddy": "Category:Teddy (dog)",
    "alex": "Category:Alex (dog)",
    "rronnie": "Category:Rronnie",
    "jerry-hohen-mark": "Category:Jerry von der Hohen Mark",
    "ulk": "Category:Ulk (dog)",
    "clipper-kennedy": "Category:Clipper (dog)",
    "king-tut-hoover": "Category:King Tut (dog)",
    "fala-roosevelt": "Category:Fala (dog)",
    "laddie-boy-harding": "Category:Laddie Boy",
    "rob-roy-coolidge": "Category:Rob Roy (dog)",
    "paul-pry-coolidge": "Category:Paul Pry (dog)",
    "prudence-prim-coolidge": "Category:Prudence Prim (dog)",
}

# (category, max files, subcategory depth). Cats and other quadrupeds first:
# they are what a dog detector actually confuses, so they carry the most signal
# about the false-positive rate. "Quality images" categories are curated
# photographs, which keeps diagrams and scans out without extra filtering.
NEGATIVE_CATEGORIES = [
    ("Category:Quality images of cats", 110, 2),
    ("Category:Featured pictures of cats", 15, 0),
    ("Category:Quality images of Canidae", 70, 1),  # wolves, foxes, jackals
    ("Category:Quality images of horses", 35, 0),
    ("Category:Quality images of cattle", 30, 0),
    ("Category:Quality images of sheep", 20, 0),
    ("Category:Quality images of goats", 20, 0),
    ("Category:Quality images of Cervidae", 20, 1),
    ("Category:Quality images of Sus scrofa", 15, 0),
    ("Category:Quality images of Oryctolagus cuniculus", 10, 0),
    ("Category:Quality images of people", 60, 0),
    ("Category:Quality images of landscapes", 35, 0),
    ("Category:Quality images of buildings", 20, 0),
    ("Category:Quality images of automobiles", 20, 0),
    ("Category:Quality images of food", 10, 0),
]

MPDD_URL = "https://data.mendeley.com/public-files/datasets/v5j6m8dzhv/files/05d1d583-faf6-410d-89a5-a6b1134b6e5e/file_downloaded"
MPDD_SHA256 = "6c800c1b4aa67629544dec7444dee85bc57781abde1ae1d077ea9ef1804284cd"
MPDD_LICENCE = "CC BY 4.0"
MPDD_CREDIT = "Multi-pose Dog Dataset, doi:10.17632/v5j6m8dzhv.1"

# Commons categories are full of things that are not photographs of the dog:
# statues, graves, stamps, paintings, merchandise, and — for the historic
# presidential dogs — archive contact sheets, which are grids of tiny frames.
NON_PHOTO = re.compile(
    r"statue|monument|memorial|denkmal|estatua|standbeeld|grave|headstone|tomb|"
    r"plaque|bust\b|sculpt|bronze|marble|relief|figurine|replica|model of|"
    r"stamp|coin|medal|\bawards?\b|banknote|logo|poster|placard|banner|badge|"
    r"postcard|\bcards?\b|cartoon|comic|drawing|painting|engrav|lithograph|etching|"
    r"illustrat|sketch|woodcut|portrait of|artwork|mural|graffiti|street art|"
    r"map of|diagram|chart|book|cover|letter|document|manuscript|newspaper|"
    r"contact sheet|exhibit|toy|plush|screenshot|collar|leash|kennel",
    re.I,
)

# Narrower, because a file's own categories are noisier than its title: a photo
# taken in a museum is fine, a photo *of* a sculpture is not.
NON_PHOTO_CATEGORY = re.compile(
    r"statues|sculptures|monuments|memorials|graves|paintings|drawings|"
    r"engravings|lithographs|illustrations|cartoons|stamps|coins|logos|"
    r"posters|contact sheets|taxidermy|comics|artworks",
    re.I,
)

# Applied to negatives only: a "cat" or "landscape" photo with a dog in it is
# not a negative.
DOG_WORDS = re.compile(
    r"\bdogs?\b|\bpupp|\bhunde?\b|\bchien|\bperro|\bhond\b|\bcane\b|\bcão\b|"
    r"familiaris|terrier|retriever|spaniel|poodle|collie|shepherd dog|"
    r"sheepdog|husky|dachshund|greyhound|\bhound\b|mastiff|chihuahua|corgi",
    re.I,
)


def api(**params) -> dict:
    """One Commons API call, retrying through their rate limiter."""
    params |= {"format": "json", "formatversion": "2"}
    for attempt in range(6):
        time.sleep(0.3)
        try:
            r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    return {}


def category_files(category: str, depth: int = 0) -> list[str]:
    """File titles in a category, walking `depth` levels of subcategories."""
    titles, cont = [], {}
    while True:
        data = api(
            action="query",
            list="categorymembers",
            cmtitle=category,
            cmtype="file",
            cmlimit=500,
            **cont,
        )
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        cont = data.get("continue", {})
        if not cont:
            break
    if depth:
        subs = api(
            action="query",
            list="categorymembers",
            cmtitle=category,
            cmtype="subcat",
            cmlimit=500,
        )
        for sub in subs.get("query", {}).get("categorymembers", []):
            titles += category_files(sub["title"], depth - 1)
    return titles


def file_info(titles: list[str]) -> dict[str, dict]:
    """Thumbnail URL, dimensions, licence and own-categories, 40 titles a call."""
    out = {}
    for i in range(0, len(titles), 40):
        data = api(
            action="query",
            prop="imageinfo",
            iiprop="url|size|mime|extmetadata",
            iiurlwidth=1024,
            titles="|".join(titles[i : i + 40]),
        )
        for page in data.get("query", {}).get("pages", []):
            info = page.get("imageinfo")
            if info:
                out[page["title"]] = info[0]
    return out


def is_photo(title: str, info: dict) -> bool:
    """Reject non-photographs by filename, own categories, format and size."""
    categories = (
        info.get("extmetadata", {})
        .get("Categories", {})
        .get("value", "")
        .replace("|", " ")
    )
    return (
        not NON_PHOTO.search(title)
        and not NON_PHOTO_CATEGORY.search(categories)
        and info.get("mime") in ("image/jpeg", "image/png")
        and min(info.get("width", 0), info.get("height", 0)) >= 400
    )


def record(title: str, info: dict, path: Path, label: str, category: str) -> dict:
    meta = info.get("extmetadata", {})
    artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
    return {
        "path": str(path.relative_to(OUT_DIR)),
        "label": label,
        "source": "commons",
        "source_url": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
        "licence": meta.get("LicenseShortName", {}).get("value", "unknown"),
        "credit": artist,
        "category": category,
    }


def slugify(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def filename(title: str) -> str:
    """Readable but unique: Commons titles collide once truncated, and a
    collision silently overwrites a photo while both stay in the manifest."""
    digest = hashlib.sha1(title.encode()).hexdigest()[:8]
    return f"{slugify(title[5:-4])[:70]}-{digest}.jpg"


def download(url: str, dest: Path) -> bool:
    """Fetch to dest unless it is already there. Returns False on a bad body.

    upload.wikimedia.org rate-limits harder than the API does, and a 429 here
    silently costs an identity a third of its photos, so back off and retry.
    """
    if dest.exists():
        return True
    for attempt in range(4):
        time.sleep(0.3)
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
            r.raise_for_status()
        except requests.RequestException as exc:
            if attempt == 3:
                logger.warning(f"  {dest.name}: {exc}")
                return False
            time.sleep(2**attempt)
            continue
        if not r.content.startswith((b"\xff\xd8", b"\x89PNG")):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    return False


def fetch_identities(limit: int | None, per_identity: int) -> list[dict]:
    records = []
    for slug, category in list(IDENTITY_CATEGORIES.items())[:limit]:
        titles = category_files(category)
        infos = file_info(titles)
        keep = [t for t in titles if t in infos and is_photo(t, infos[t])]
        if len(keep) < MIN_PHOTOS:
            logger.warning(f"{slug}: only {len(keep)} photos, skipping")
            continue
        got = []
        for title in keep[:per_identity]:
            info = infos[title]
            dest = OUT_DIR / "identities" / slug / filename(title)
            if download(info.get("thumburl") or info["url"], dest):
                got.append(record(title, info, dest, slug, category))
        if len(got) < MIN_PHOTOS:
            logger.warning(f"{slug}: only {len(got)} downloaded, skipping")
            continue
        records += got
        logger.info(f"{slug}: {len(got)} photos")
    return records


def fetch_negatives(limit: int) -> list[dict]:
    records, seen = [], set()
    per_category = {c: n for c, n, _ in NEGATIVE_CATEGORIES}
    for category, cap, depth in NEGATIVE_CATEGORIES:
        if len(records) >= limit:
            break
        titles = [t for t in category_files(category, depth) if t not in seen]
        titles = [t for t in titles if not DOG_WORDS.search(t)]
        infos = file_info(titles[: cap * 4])  # headroom for the ones we reject
        taken = 0
        for title in titles:
            if taken >= per_category[category] or len(records) >= limit:
                break
            info = infos.get(title)
            if not info or not is_photo(title, info):
                continue
            meta = info.get("extmetadata", {}).get("Categories", {}).get("value", "")
            if DOG_WORDS.search(meta.replace("|", " ")):
                continue
            seen.add(title)
            dest = OUT_DIR / "negatives" / filename(title)
            if download(info.get("thumburl") or info["url"], dest):
                records.append(record(title, info, dest, "negative", category))
                taken += 1
        logger.info(f"{category}: {taken} negatives")
    return records


def fetch_mpdd(cache: Path, limit: int | None) -> list[dict]:
    """Extract the Multi-pose Dog Dataset, grouped by its identity prefix."""
    archive = cache / "MPDD.zip"
    if not archive.exists():
        logger.info("Downloading MPDD (30 MB)...")
        archive.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(MPDD_URL, headers={"User-Agent": UA}, timeout=600)
        r.raise_for_status()
        archive.write_bytes(r.content)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != MPDD_SHA256:
        logger.error(f"MPDD checksum mismatch ({digest}), skipping")
        return []

    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as zf:
        by_identity = defaultdict(list)
        for name in zf.namelist():
            if name.lower().endswith(".jpg"):
                by_identity[Path(name).name.split("_")[0]].append(name)
        usable = {k: v for k, v in sorted(by_identity.items()) if len(v) >= MIN_PHOTOS}
        records = []
        for identity, names in list(usable.items())[:limit]:
            slug = f"mpdd-{int(identity):04d}"
            for name in names:
                dest = OUT_DIR / "identities" / slug / Path(name).name
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(name))
                records.append(
                    {
                        "path": str(dest.relative_to(OUT_DIR)),
                        "label": slug,
                        "source": "mpdd",
                        "source_url": "https://doi.org/10.17632/v5j6m8dzhv.1",
                        "licence": MPDD_LICENCE,
                        "credit": MPDD_CREDIT,
                        "category": "MPDD",
                    }
                )
    logger.info(f"mpdd: {len(usable)} identities, {len(records)} photos")
    return records


def summarise(records: list[dict]) -> None:
    identities = defaultdict(list)
    negatives = 0
    for r in records:
        if r["label"] == "negative":
            negatives += 1
        else:
            identities[r["label"]].append(r)
    by_source = defaultdict(int)
    for label, rows in identities.items():
        by_source[rows[0]["source"]] += 1
    counts = sorted(len(v) for v in identities.values())
    print(f"\nidentities : {len(identities)} ({dict(by_source)})")
    print(f"photos     : {sum(counts)} (min {counts[0]}, max {counts[-1]})")
    print(f"negatives  : {negatives}")
    print(f"manifest   : {OUT_DIR / 'manifest.json'}\n")
    for label, rows in sorted(identities.items()):
        if rows[0]["source"] == "commons":
            print(f"  {label:<42} {len(rows):>3}")


def main(args: argparse.Namespace) -> None:
    global OUT_DIR  # every path is built from it, and --out has to reach them all
    OUT_DIR = Path(args.out)
    records = fetch_identities(args.limit, args.per_identity)
    if not args.no_mpdd:
        records += fetch_mpdd(OUT_DIR / ".cache", args.limit or args.mpdd_identities)
    records += fetch_negatives(args.limit or args.negatives)

    # Drop entries whose file vanished, so the manifest always matches disk.
    records = [r for r in records if (OUT_DIR / r["path"]).exists()]
    (OUT_DIR / "manifest.json").write_text(
        json.dumps({"files": records}, indent=2, ensure_ascii=False)
    )
    summarise(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="smoke run: at most N identities and N negatives",
    )
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--per-identity", type=int, default=30)
    parser.add_argument("--negatives", type=int, default=450)
    parser.add_argument("--mpdd-identities", type=int, default=60)
    parser.add_argument("--no-mpdd", action="store_true")
    main(parser.parse_args())
