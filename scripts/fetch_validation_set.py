#!/usr/bin/env python
"""Build a held-out validation set for the Immich sidecar.

Two things the sidecar has to get right, and neither is measurable on the
training splits: does it find the dog in a photo, and does it *avoid* firing on
photos with no dog (every false positive becomes a junk "person" in Immich).

    data/sidecar-validation/
      identities/<slug>/*.jpg   photos of one individual dog (or cat-<slug>: cat)
      negatives/*.jpg           photos with no dog at all
      manifest.json             every file, its label, species, source URL and licence

Sources: hand-picked Wikimedia Commons categories for named individual dogs and
cats (free licences, but noisy — see NON_PHOTO), Commons "Quality images" categories
for negatives, and the embedder's MPDD test identities (``data/mpdd``) for a
second opinion on identity. MPDD images are reID-style crops, not snapshots, so
the manifest tags every file with its source and the evaluator scores them apart.

    uv run python scripts/fetch_validation_set.py
    uv run python scripts/fetch_validation_set.py --limit 3   # smoke run
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

from animal_id.common.constants import DATA_DIR, PROJECT_ROOT
from animal_id.common.logging_config import setup_logging
from animal_id.data import sources
from animal_id.data.exports import torch_identity
from animal_id.data.sample import Source
from animal_id.data.sources import mpdd

logger = setup_logging(__name__, logging.INFO)

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

# Same selection rules, from Category:Individual cats / Famous cats / Cats of
# Wikipedians. Groups ("Nikita and Mika"), pairs, fictional cats, mascots and
# statue-only categories are out.
CAT_CATEGORIES = {
    "cat-alpha": "Category:Alpha (cat)",
    "cat-bella": "Category:Bella (cat)",
    "cat-caramel": "Category:Caramel (red tabby cat)",
    "cat-chocolate-bangkok": "Category:Chocolate Bangkok cat",
    "cat-cola": "Category:Cola (solid black cat)",
    "cat-combat": "Category:Combat (cat)",
    "cat-ellie": "Category:Ellie (cat)",
    "cat-erik-skaelskor": "Category:Erik the cat (Skælskør)",
    "cat-fiumi": "Category:Fiumi (cat)",
    "cat-foss": "Category:Foss (cat)",
    "cat-francine": "Category:Francine (cat)",
    "cat-george-de-jolival": "Category:George de Jolival",
    "cat-gillie": "Category:Gillie (cat)",
    "cat-gladstone": "Category:Gladstone (cat)",
    "cat-grumpy": "Category:Grumpy Cat",
    "cat-india-bush": "Category:India (cat)",
    "cat-jule": "Category:Jule (female tuxedo patterned red bloched tabby and white cat)",
    "cat-kater-stanislaus": "Category:Kater Stanislaus",
    "cat-larry": "Category:Larry the cat",
    "cat-leerie": "Category:Leerie (Asian cat)",
    "cat-loki": "Category:Loki (cat)",
    "cat-mikan-station": "Category:Mikan (cat)",
    "cat-miss-kitty": "Category:Miss Kitty (black silver cat)",
    "cat-miss-stevie-nicks": "Category:Miss Stevie Nicks (cat)",
    "cat-miss-truffles": "Category:Miss Truffles (cat)",
    "cat-miyako": "Category:Miyako (cat)",
    "cat-ms-squeaky": "Category:Ms. Squeaky (cat)",
    "cat-nanouk": "Category:Nanouk (cat)",
    "cat-naro": "Category:Naro (cat)",
    "cat-oiseau": "Category:Oiseau (cat)",
    "cat-onapromise-pyanfar": "Category:Onapromise Pyanfar (cat)",
    "cat-osiris": "Category:Osiris (Bengal cat)",
    "cat-palmerston": "Category:Palmerston (cat)",
    "cat-peach": "Category:Peach (red tabby and white cat)",
    "cat-poseidon": "Category:Poseidon (cat)",
    "cat-prickles": "Category:Prickles (Asian cat)",
    "cat-rambo": "Category:Rambo (cat)",
    "cat-romeo": "Category:Romeo (male tuxedo patterned red blotched tabby and white cat)",
    "cat-rum-tum-tugger": "Category:Rum Tum Tugger (red tabby cat)",
    "cat-samourai": "Category:Samouraï (cat)",
    "cat-socks-clinton": "Category:Socks (cat)",
    "cat-stephanie": "Category:Stephanie (cat)",
    "cat-suzieq": "Category:SuzieQ (cat)",
    "cat-tama-station": "Category:Tama (cat)",
    "cat-tesla": "Category:Tesla (Ragdoll cat)",
    "cat-tomcat-matata": "Category:Tomcat Matata (cat)",
    "cat-uthello": "Category:Uthello (Sphynx cat)",
    "cat-vikingur": "Category:Víkingur (NFO cat)",
    "cat-willow-biden": "Category:Willow (Joe Biden's cat)",
    "cat-tiger-coolidge": "Category:Tiger (pet cat of Grace Coolidge)",
    "cat-misty-malarky-ying-yang": "Category:Misty Malarky Ying Yang",
    "cat-trim": "Category:Trim (cat)",
    "cat-mrs-chippy": "Category:Mrs. Chippy",
}

# Domestic-cat negatives stay labelled "negative" so dog scoring is unchanged,
# but are tagged species "cat" for cat scoring.
CAT_NEGATIVE_CATEGORIES = {
    "Category:Quality images of cats",
    "Category:Featured pictures of cats",
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


def source_url(title: str) -> str:
    return f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"


def record(
    title: str,
    info: dict,
    path: Path,
    label: str,
    category: str,
    species: str | None,
) -> dict:
    meta = info.get("extmetadata", {})
    artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
    return {
        "path": str(path.relative_to(OUT_DIR)),
        "label": label,
        "species": species,
        "source": "commons",
        "source_url": source_url(title),
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


def fetch_identities(
    categories: dict[str, str], species: str, limit: int | None, per_identity: int
) -> list[dict]:
    records = []
    for slug, category in list(categories.items())[:limit]:
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
                got.append(record(title, info, dest, slug, category, species))
        if len(got) < MIN_PHOTOS:
            logger.warning(f"{slug}: only {len(got)} downloaded, skipping")
            continue
        records += got
        logger.info(f"{slug}: {len(got)} photos")
    return records


def fetch_negatives(limit: int, identity_urls: set[str]) -> list[dict]:
    records, seen = [], set()
    per_category = {c: n for c, n, _ in NEGATIVE_CATEGORIES}
    for category, cap, depth in NEGATIVE_CATEGORIES:
        if len(records) >= limit:
            break
        titles = [t for t in category_files(category, depth) if t not in seen]
        titles = [t for t in titles if not DOG_WORDS.search(t)]
        titles = [t for t in titles if source_url(t) not in identity_urls]
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
                species = "cat" if category in CAT_NEGATIVE_CATEGORIES else None
                records.append(record(title, info, dest, "negative", category, species))
                taken += 1
        logger.info(f"{category}: {taken} negatives")
    return records


def fetch_mpdd(limit: int | None) -> list[dict]:
    """MPDD's embedder test identities, so the sidecar is scored on dogs it never trained on."""
    by_identity = defaultdict(list)
    for row in torch_identity.splits(sources.load(Source.MPDD))["test"]:
        by_identity[row["identity_label"]].append(PROJECT_ROOT / row["file_path"])
    records = []
    for paths in list(by_identity.values())[:limit]:
        slug = f"mpdd-{int(paths[0].name.split('_')[0]):04d}"
        for path in paths:
            dest = OUT_DIR / "identities" / slug / path.name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
            records.append(
                {
                    "path": str(dest.relative_to(OUT_DIR)),
                    "label": slug,
                    "species": "dog",
                    "source": "mpdd",
                    "source_url": "https://doi.org/10.17632/v5j6m8dzhv.1",
                    "licence": mpdd.LICENSE,
                    "credit": "Multi-pose Dog Dataset, doi:10.17632/v5j6m8dzhv.1",
                    "category": "MPDD",
                }
            )
    logger.info(f"mpdd: {len(by_identity)} test identities, {len(records)} photos")
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
    records = fetch_identities(
        IDENTITY_CATEGORIES, "dog", args.limit, args.per_identity
    )
    cats = fetch_identities(CAT_CATEGORIES, "cat", args.limit, args.per_identity)
    # A photo filed under two cats shows both, so it labels neither.
    shared = {u for u, n in Counter(r["source_url"] for r in cats).items() if n > 1}
    records += [r for r in cats if r["source_url"] not in shared]
    if not args.no_mpdd:
        records += fetch_mpdd(args.limit)
    identity_urls = {r["source_url"] for r in records}
    records += fetch_negatives(args.limit or args.negatives, identity_urls)

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
    parser.add_argument("--no-mpdd", action="store_true")
    main(parser.parse_args())
