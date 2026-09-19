"""Report the face edits that Face Detection → Reset would discard.

Refresh keeps all of it; this is the cost of the Reset button.

Usage: python3 audit_face_edits.py [--container immich_postgres] [--sql]
"""

import argparse
import subprocess
import sys

# Immich v3 keys people by personGroupId; every version before it keys them by
# person.id, and only v3 can hide an individual face.
PROBE = """
select
  (select count(*) from information_schema.columns
    where table_name = 'person' and column_name = 'personGroupId'),
  (select count(*) from information_schema.columns
    where table_name = 'asset_face' and column_name = 'isVisible')
"""

# Every count here is a column, never an inference. Merges and reassignments are
# left out because Immich stores no evidence of them: a merge deletes the losing
# person and a reassignment only moves a face to another group.
QUERY = """
select
  (select count(*) from person where name <> ''),
  (select count(*) from asset_face af
     join person p on p."{person_key}" = af."{face_key}"
    where p.name <> '' and af."deletedAt" is null
      and not exists (
        select 1 from asset_face k
         where k."{face_key}" = p."{person_key}"
           and k."sourceType" <> 'machine-learning')),
  (select count(*) from person p
    where p.name <> '' and exists (
      select 1 from asset_face af
       where af."{face_key}" = p."{person_key}"
         and af."sourceType" <> 'machine-learning')),
  (select count(*) from person where "isHidden"),
  (select count(*) from person where "isFavorite"),
  (select count(*) from person where "birthDate" is not null),
  {hidden_faces},
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


def psql(container: str, user: str, database: str, sql: str) -> list[str]:
    command = ["docker", "exec", "-i", container, "psql", "-U", user, "-d", database]
    result = subprocess.run(
        [*command, "-At", "-F", "\t", "-c", sql], capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(result.stderr.strip() or f"could not query {container}")
    return result.stdout.strip().split("\t")


def build_query(container: str, user: str, database: str) -> str:
    groups, visible = psql(container, user, database, PROBE)
    return QUERY.format(
        person_key="personGroupId" if groups == "1" else "id",
        face_key="personGroupId" if groups == "1" else "personId",
        hidden_faces=(
            '(select count(*) from asset_face where not "isVisible")'
            if visible == "1"
            else "null"
        ),
    )


def run(container: str, user: str, database: str) -> dict[str, int | None]:
    values = psql(container, user, database, build_query(container, user, database))
    return {k: int(v) if v else None for k, v in zip(FIELDS, values, strict=True)}


def report(c: dict[str, int | None]) -> None:
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

    # A count is None where this Immich version has no such column to read.
    lost = [row for row in lost if row[1] is not None]

    print(f"\n{c['people']} people, {c['faces']} faces.\n")
    if not any(n for _, n, _ in lost + kept):
        print(f"No face edits found. Nothing to lose.\n\n{UNCOUNTED}\n")
        return

    print("Lost if you Reset Face Detection")
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
        print(build_query(args.container, args.user, args.database).strip())
        return
    report(run(args.container, args.user, args.database))


if __name__ == "__main__":
    main()
