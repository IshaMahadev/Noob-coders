#!/usr/bin/env python3
"""Streaming cross-checks for the large output TSVs.

The challenge-provided validator should still be run on matching_results.tsv.
This helper checks candidate/match row alignment and subset consistency without
loading the multi-million-row candidate map into memory.
"""

import argparse
import csv
from pathlib import Path


def ids(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    rows = 0
    with args.matching.open(encoding="utf-8", newline="") as mf, args.candidate.open(encoding="utf-8", newline="") as cf:
        mr, cr = csv.reader(mf, delimiter="\t"), csv.reader(cf, delimiter="\t")
        if next(mr, None) != ["source1_entity_id", "matched_entity_ids"]:
            raise SystemExit("FAIL: matching header is incorrect")
        if next(cr, None) != ["source1_entity_id", "candidate_entity_ids"]:
            raise SystemExit("FAIL: candidate header is incorrect")
        while True:
            mrow, crow = next(mr, None), next(cr, None)
            if mrow is None or crow is None:
                if mrow is not None or crow is not None:
                    raise SystemExit("FAIL: matching and candidate row counts differ")
                break
            rows += 1
            if len(mrow) != 2 or len(crow) != 2 or mrow[0] != crow[0]:
                raise SystemExit(f"FAIL: malformed or misaligned rows at output row {rows + 1}")
            matched, candidates = ids(mrow[1]), ids(crow[1])
            if len(matched) != len(set(matched)) or len(candidates) != len(set(candidates)):
                raise SystemExit(f"FAIL: duplicate target ID at Source 1 {mrow[0]}")
            if any(not x.startswith(("S2-", "S3-")) for x in candidates + matched):
                raise SystemExit(f"FAIL: invalid target ID prefix at Source 1 {mrow[0]}")
            if not set(matched).issubset(candidates):
                raise SystemExit(f"FAIL: final match absent from candidate list at Source 1 {mrow[0]}")
    if rows != args.expected_rows:
        raise SystemExit(f"FAIL: expected {args.expected_rows:,} rows; found {rows:,}")
    print(f"PASS: {rows:,} aligned rows; no duplicates, bad prefixes, or non-candidate matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
