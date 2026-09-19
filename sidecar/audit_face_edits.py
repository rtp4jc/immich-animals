"""Report the face edits that re-running Face Detection would discard.

Run it before pointing Immich at the sidecar, and again before turning it off.

Usage: python3 audit_face_edits.py [--container immich_postgres] [--sql]
"""

import argparse
import subprocess
import sys

# Immich prunes the *_audit tables after 31 days (sync.service.js MAX_DAYS), so
# merges older than that leave no trace at all.
AUDIT_DAYS = 31

# A job run rewrites a large share of the library in one minute; a person edits a
# handful. 1% separates the two on every library we have measured.
#
# Edits made before the last mass face deletion are already gone, so `cutoff`
# finds that job run and the estimates below only count what came after it.
QUERY = """
with cutoff as (
  select coalesce(max(m), '-infinity'::timestamptz) c
    from (select date_trunc('minute', "deletedAt") m, count(*) n
            from asset_face_audit group by 1) s,
         (select greatest(count(*), 1) * 0.01 t from asset_face) k
   where n >= t)
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
  (select count(*) from asset_face where "sourceType" <> 'machine-learning'),
  (select count(*) from asset_face where not "isVisible"),
  (select count(*) from asset_face where "deletedAt" is not null),
  (select coalesce(sum(n) filter (where n < t), 0)
     from (select date_trunc('minute', "updatedAt") m, count(*) n
             from asset_face where "updatedAt" > cutoff.c group by 1) s,
          (select greatest(count(*), 1) * 0.01 t from asset_face) k),
  (select coalesce(sum(n) filter (where n < t), 0)
     from (select date_trunc('minute', "deletedAt") m, count(*) n
             from person_audit where "deletedAt" > cutoff.c group by 1) s,
          (select greatest(count(*), 1) * 0.01 t from person) k),
  (select count(*) from person),
  (select count(*) from asset_face where "deletedAt" is null)
from cutoff
"""

FIELDS = (
    "named named_faces named_surviving hidden favorite birthdate manual_faces "
    "hidden_faces deleted_faces hand_faces merges people faces"
).split()


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
        k: int(float(v))
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
        ("merges", c["merges"], f"estimated, last {AUDIT_DAYS} days"),
        ("hand edits to faces", c["hand_faces"], "estimated: moved, hidden or deleted"),
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
        print("No manual face edits found. Nothing to lose.\n")
        return

    print("Lost when you re-run Face Detection")
    for label, n, note in lost:
        print(f"  {n:>6}  {label}{f'  ({note})' if note and n else ''}")
    if c["deleted_faces"]:
        print(f"  {c['deleted_faces']:>6}  deleted faces  (these come back)")
    print("\nKept")
    for label, n, note in kept:
        print(f"  {n:>6}  {label}{f'  ({note})' if note and n else ''}")
    print()


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
