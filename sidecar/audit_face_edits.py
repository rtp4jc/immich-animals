"""Report the face edits that re-running Face Detection would discard.

Run it before pointing Immich at the sidecar, and again before turning it off.

Usage: python3 audit_face_edits.py [--container immich_postgres] [--sql]
"""

import argparse
import subprocess
import sys

# Every count here is a column, never an inference. Merges and reassignments are
# left out because Immich stores no evidence of them: a merge deletes the losing
# person and a reassignment only moves a face to another group.
QUERY = """
select
  (select count(*) from person where name <> ''),
  (select count(*) from asset_face af
     join person p on p."personGroupId" = af."personGroupId"
    where p.name <> '' and af."deletedAt" is null),
  (select count(*) from person p
    where p.name <> '' and exists (
      select 1 from asset_face af
       where af."personGroupId" = p."personGroupId"
         and af."sourceType" <> 'machine-learning')),
  (select count(*) from person where "isHidden"),
  (select count(*) from person where "isFavorite"),
  (select count(*) from person where "birthDate" is not null),
  (select count(*) from asset_face where not "isVisible"),
  (select count(*) from asset_face where "deletedAt" is not null),
  (select count(*) from asset_face where "sourceType" <> 'machine-learning'),
  (select count(*) from person),
  (select count(*) from asset_face where "deletedAt" is null)
"""

FIELDS = (
    "named named_faces named_surviving hidden favorite birthdate "
    "hidden_faces deleted_faces manual_faces people faces"
).split()

UNCOUNTED = """Merges and faces moved between people are missing from this list:
Immich records neither, so no tool can count them. They go the same way."""


def run(container: str, user: str, database: str) -> dict[str, int]:
    result = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            container,
            "psql",
            "-U",
            user,
            "-d",
            database,
            "-At",
            "-F",
            "\t",
            "-c",
            QUERY,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(result.stderr.strip() or f"could not query {container}")
    return {
        k: int(v)
        for k, v in zip(FIELDS, result.stdout.strip().split("\t"), strict=True)
    }


def report(c: dict[str, int]) -> None:
    lost = [
        (
            "named people",
            c["named"] - c["named_surviving"],
            f"{c['named_faces']} faces between them",
        ),
        ("hidden people", c["hidden"], ""),
        ("favourited people", c["favorite"], ""),
        ("birth dates", c["birthdate"], ""),
        ("hidden faces", c["hidden_faces"], ""),
        ("deleted faces", c["deleted_faces"], "these come back"),
    ]
    kept = [
        ("manually added faces", c["manual_faces"], ""),
        (
            "named people holding one",
            c["named_surviving"],
            "they keep the name, but lose every detected face",
        ),
    ]

    print(f"\n{c['people']} people, {c['faces']} faces.\n")
    if not any(n for _, n, _ in lost + kept):
        print(f"No face edits found. Nothing to lose.\n\n{UNCOUNTED}\n")
        return

    print("Lost when you re-run Face Detection")
    for label, n, note in lost:
        print(f"  {n:>6}  {label}{f'  ({note})' if note and n else ''}")
    print("\nKept")
    for label, n, note in kept:
        print(f"  {n:>6}  {label}{f'  ({note})' if note and n else ''}")
    print(f"\n{UNCOUNTED}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--container", default="immich_postgres")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--database", default="immich")
    parser.add_argument(
        "--sql", action="store_true", help="print the query instead of running it"
    )
    args = parser.parse_args()

    if args.sql:
        print(QUERY.strip())
        return
    report(run(args.container, args.user, args.database))


if __name__ == "__main__":
    main()
