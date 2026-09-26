#!/usr/bin/env python3
"""Baseline business entity resolution pipeline for Amazon ML Challenge 2026.

Uses only the supplied TSVs and the Python standard library. Candidate blocking
is an inverted token index stored in SQLite so the complete source tables do not
need to fit in memory. See the project README for commands and limitations.
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable

csv.field_size_limit(2**20)

LEGAL = {
    "inc", "incorporated", "corp", "corporation", "company", "co", "llc",
    "ltd", "limited", "plc", "private", "pvt", "llp", "gmbh", "sas", "sarl",
}
ABBREVIATIONS = {
    " rd ": " road ", " st ": " street ", " ave ": " avenue ",
    " blvd ": " boulevard ", " hwy ": " highway ", " pvt ": " private ",
    " corp ": " corporation ", " ltd ": " limited ", " co ": " company ",
    "&": " and ",
}
TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)
_WORKER_DB: sqlite3.Connection | None = None


def normalize(value: str) -> str:
    value = (value or "").casefold().replace("&", " and ")
    value = TOKEN_RE.sub(" ", value)
    words = value.split()
    # Remove legal suffixes only at the end; interior words can be meaningful.
    while words and words[-1] in LEGAL:
        words.pop()
    return " ".join(words)


def tokens(record: dict[str, str]) -> tuple[list[str], list[str]]:
    name = normalize(record.get("business_name", ""))
    address = normalize(record.get("business_address", ""))
    name_tokens = list(dict.fromkeys(t for t in name.split() if len(t) >= 2))
    address_tokens = list(dict.fromkeys(t for t in address.split() if len(t) >= 3))
    return name_tokens, address_tokens


def stable_holdout(entity_id: str, percent: int) -> bool:
    value = int(hashlib.blake2b(entity_id.encode(), digest_size=4).hexdigest(), 16)
    return value % 100 < percent


def read_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def source_paths(data_dir: Path, split: str) -> tuple[Path, Path, Path]:
    folder = data_dir / split
    prefix = "train" if split == "train" else "test"
    return tuple(folder / f"{prefix}_source{i}.tsv" for i in (1, 2, 3))  # type: ignore[return-value]


def make_database(db_path: Path, data_dir: Path, split: str, holdout_percent: int = 0) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-100000")  # approx 100 MB
    db.executescript("""
      CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE records (
        entity_id TEXT PRIMARY KEY, source INTEGER NOT NULL,
        name TEXT NOT NULL, address TEXT NOT NULL, country TEXT NOT NULL,
        name_norm TEXT NOT NULL, address_norm TEXT NOT NULL
      );
      CREATE TABLE postings (
        source INTEGER NOT NULL, token TEXT NOT NULL, entity_id TEXT NOT NULL
      );
      CREATE TABLE token_freq (token TEXT NOT NULL, source INTEGER NOT NULL, n INTEGER NOT NULL,
                               PRIMARY KEY(token, source));
    """)
    db.execute("INSERT INTO metadata VALUES ('split',?)", (split,))
    s1_path, s2_path, s3_path = source_paths(data_dir, split)
    # In validation mode, use a deterministic, labeled holdout as the query set;
    # candidate records remain the corresponding supplied source tables.
    for source, path in ((1, s1_path), (2, s2_path), (3, s3_path)):
        print(f"Indexing Source {source}: {path}", flush=True)
        record_batch: list[tuple] = []
        post_batch: list[tuple] = []
        count = 0
        for row in read_tsv(path):
            entity_id = row["entity_id"]
            if source == 1 and holdout_percent and not stable_holdout(entity_id, holdout_percent):
                continue
            name, address = row.get("business_name", ""), row.get("business_address", "")
            name_norm, address_norm = normalize(name), normalize(address)
            record_batch.append((entity_id, source, name, address,
                                row.get("country", ""), name_norm, address_norm))
            nt, at = tokens(row)
            post_batch.extend((source, tok, entity_id) for tok in set(nt + at))
            count += 1
            if len(record_batch) >= 5000:
                db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", record_batch)
                db.executemany("INSERT INTO postings VALUES (?,?,?)", post_batch)
                record_batch.clear(); post_batch.clear()
                if count % 100000 == 0:
                    print(f"  indexed {count:,}", flush=True)
        if record_batch:
            db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?)", record_batch)
            db.executemany("INSERT INTO postings VALUES (?,?,?)", post_batch)
        db.commit()
        print(f"  indexed total {count:,}", flush=True)
    print("Building token lookup indexes (this may take a while)...", flush=True)
    db.execute("CREATE INDEX postings_token_source ON postings(token, source)")
    db.execute("CREATE INDEX postings_entity ON postings(entity_id)")
    db.execute("CREATE INDEX records_source ON records(source)")
    print("Counting token frequencies...", flush=True)
    # Match the postings index order to avoid a second large sort on the corpus.
    db.execute("INSERT INTO token_freq SELECT token, source, COUNT(*) FROM postings GROUP BY token, source")
    db.commit()
    db.close()


def load_truth(path: Path, holdout_percent: int = 0) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    for row in read_tsv(path):
        s1 = row["source1_entity_id"]
        if holdout_percent and not stable_holdout(s1, holdout_percent):
            continue
        truth[s1] = {x.strip() for x in row.get("matched_entity_ids", "").split(",") if x.strip()}
    return truth


def candidate_features(s1: dict[str, str], candidate: sqlite3.Row) -> tuple[float, float, float, float]:
    a_name, a_addr = set(normalize(s1.get("business_name", "")).split()), set(normalize(s1.get("business_address", "")).split())
    b_name, b_addr = set(candidate["name_norm"].split()), set(candidate["address_norm"].split())
    name_j = len(a_name & b_name) / max(1, len(a_name | b_name))
    addr_j = len(a_addr & b_addr) / max(1, len(a_addr | b_addr))
    name_contain = len(a_name & b_name) / max(1, min(len(a_name), len(b_name)))
    addr_contain = len(a_addr & b_addr) / max(1, min(len(a_addr), len(b_addr)))
    return name_j, addr_j, name_contain, addr_contain


def retrieve(db: sqlite3.Connection, row: dict[str, str], max_per_source: int,
             max_df: int) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    nt, at = tokens(row)
    query_tokens = list(dict.fromkeys(nt + at))
    if not query_tokens:
        return [], []
    placeholders = ",".join("?" for _ in query_tokens)
    # Very common tokens create huge, uninformative blocks. Discard them using
    # corpus document frequency before fetching candidates.
    ranked = db.execute(
        f"WITH overlaps AS (SELECT p.entity_id,p.source,COUNT(*) overlap "
        f"FROM postings p JOIN token_freq f ON f.source=p.source AND f.token=p.token "
        f"WHERE p.token IN ({placeholders}) AND p.source IN (2,3) AND f.n <= ? "
        "GROUP BY p.entity_id,p.source), ranked AS ("
        "SELECT entity_id,source,overlap,ROW_NUMBER() OVER (PARTITION BY source ORDER BY overlap DESC,entity_id) rn FROM overlaps) "
        "SELECT r.entity_id,r.source,r.name_norm,r.address_norm,r.country,ranked.overlap "
        "FROM ranked JOIN records r USING(entity_id) WHERE ranked.rn <= ?",
        (*query_tokens, max_df, max_per_source * 4),
    ).fetchall()
    # Re-score name/address overlap and retain a broad shortlist per source.
    lists: dict[int, list[tuple[str, float]]] = {2: [], 3: []}
    for candidate in ranked:
        nj, aj, nc, ac = candidate_features(row, candidate)
        country_bonus = 0.08 if row.get("country", "") and row.get("country", "") == candidate["country"] else 0.0
        score = 0.58 * nj + 0.30 * aj + 0.07 * nc + 0.05 * ac + country_bonus
        lists[candidate["source"]].append((candidate["entity_id"], score))
    for source in (2, 3):
        lists[source].sort(key=lambda x: x[1], reverse=True)
        lists[source] = lists[source][:max_per_source]
    return lists[2], lists[3]


def retrieve_batch(db: sqlite3.Connection, batch: list[dict[str, str]], max_per_source: int,
                   max_df: int) -> dict[str, list[tuple[str, float]]]:
    """Retrieve candidates for a batch using a set-based SQLite join.

    Batching avoids one expensive SQL aggregation per Source 1 row, which is too
    slow for the million-row test set.
    """
    if not batch:
        return {}
    db.execute("CREATE TEMP TABLE IF NOT EXISTS qtokens (s1 TEXT, token TEXT, PRIMARY KEY(s1,token)) WITHOUT ROWID")
    db.execute("DELETE FROM qtokens")
    query_rows: list[tuple[str, str]] = []
    by_s1 = {r["entity_id"]: r for r in batch}
    token_sets: dict[str, tuple[list[str], list[str]]] = {}
    all_tokens: set[str] = set()
    for row in batch:
        nt, at = tokens(row)
        token_sets[row["entity_id"]] = (nt, at)
        all_tokens.update(nt); all_tokens.update(at)
    frequencies: dict[str, int] = {}
    unique_tokens = sorted(all_tokens)
    # Fetch document frequencies in bounded parameter groups, then use only a
    # few rare terms from each field to keep large-corpus joins selective.
    for offset in range(0, len(unique_tokens), 800):
        chunk = unique_tokens[offset:offset + 800]
        placeholders = ",".join("?" for _ in chunk)
        freq_rows = db.execute(
            f"SELECT token,MAX(n) AS n FROM token_freq WHERE token IN ({placeholders}) GROUP BY token", chunk
        ).fetchall()
        frequencies.update((r["token"], r["n"]) for r in freq_rows)
    for s1_id, (nt, at) in token_sets.items():
        rare_name = sorted((t for t in nt if frequencies.get(t, max_df + 1) <= max_df),
                           key=lambda t: frequencies.get(t, max_df + 1))[:1]
        rare_address = sorted((t for t in at if frequencies.get(t, max_df + 1) <= max_df),
                              key=lambda t: frequencies.get(t, max_df + 1))[:1]
        query_rows.extend((s1_id, token) for token in dict.fromkeys(rare_name + rare_address))
    db.executemany("INSERT OR IGNORE INTO qtokens VALUES (?,?)", query_rows)
    ranked = db.execute(
        "WITH overlaps AS ("
        " SELECT q.s1,p.entity_id,p.source,COUNT(*) overlap"
        " FROM qtokens q JOIN token_freq f ON f.token=q.token AND f.source IN (2,3) AND f.n <= ?"
        " JOIN postings p ON p.token=q.token AND p.source=f.source"
        " GROUP BY q.s1,p.entity_id,p.source"
        "), ranked AS ("
        " SELECT s1,entity_id,source,overlap,"
        " ROW_NUMBER() OVER (PARTITION BY s1,source ORDER BY overlap DESC,entity_id) rn"
        " FROM overlaps"
        ") SELECT ranked.s1,r.entity_id,r.source,r.name_norm,r.address_norm,r.country"
        " FROM ranked JOIN records r ON r.entity_id=ranked.entity_id"
        " WHERE ranked.rn <= ?",
        (max_df, max_per_source),
    ).fetchall()
    lists: dict[str, dict[int, list[tuple[str, float]]]] = defaultdict(lambda: {2: [], 3: []})
    for candidate in ranked:
        row = by_s1[candidate["s1"]]
        nj, aj, nc, ac = candidate_features(row, candidate)
        country_bonus = 0.08 if row.get("country", "") and row.get("country", "") == candidate["country"] else 0.0
        score = 0.58 * nj + 0.30 * aj + 0.07 * nc + 0.05 * ac + country_bonus
        lists[candidate["s1"]][candidate["source"]].append((candidate["entity_id"], score))
    result: dict[str, list[tuple[str, float]]] = {}
    for s1_id in by_s1:
        choices = lists[s1_id]
        combined = []
        for source in (2, 3):
            choices[source].sort(key=lambda item: item[1], reverse=True)
            combined.extend(choices[source][:max_per_source])
        result[s1_id] = combined
    return result


def initialize_worker(db_path: str) -> None:
    global _WORKER_DB
    _WORKER_DB = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    _WORKER_DB.row_factory = sqlite3.Row


def worker_retrieve(payload: tuple[list[dict[str, str]], int, int]):
    batch, max_per_source, max_df = payload
    if _WORKER_DB is None:
        raise RuntimeError("Worker database was not initialized")
    return batch, retrieve_batch(_WORKER_DB, batch, max_per_source, max_df)


def f05(predicted: set[str], actual: set[str]) -> float:
    if not predicted and not actual:
        return 1.0
    if not predicted or not actual:
        return 0.0
    tp = len(predicted & actual)
    precision, recall = tp / len(predicted), tp / len(actual)
    denominator = 0.25 * precision + recall
    return 1.25 * precision * recall / denominator if denominator else 0.0


def run(data_dir: Path, split: str, out_dir: Path, db_path: Path, threshold: float,
        max_per_source: int, max_df: int, holdout_percent: int = 0,
        reuse_db: bool = False, batch_size: int = 500, workers: int = 1) -> None:
    if reuse_db:
        if not db_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing database: {db_path}")
        check_db = sqlite3.connect(db_path)
        try:
            indexed_split = check_db.execute("SELECT value FROM metadata WHERE key='split'").fetchone()
        except sqlite3.OperationalError as exc:
            raise ValueError("The existing database has no split metadata; rebuild it without --reuse-db") from exc
        finally:
            check_db.close()
        if not indexed_split or indexed_split[0] != split:
            raise ValueError(f"Database split is {indexed_split[0] if indexed_split else 'unknown'}, requested {split}; rebuild without --reuse-db")
    else:
        make_database(db_path, data_dir, split, holdout_percent)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    s1_path, _, _ = source_paths(data_dir, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / "candidate_pairs.tsv"
    matches_path = out_dir / "matching_results.tsv"
    truth = load_truth(data_dir / "train" / "train_ground_truth.tsv", holdout_percent) if split == "train" else {}
    scores: list[tuple[list[tuple[str, float]], set[str]]] = []
    total = 0
    with candidates_path.open("w", encoding="utf-8", newline="") as cf, matches_path.open("w", encoding="utf-8", newline="") as mf:
        cw, mw = csv.writer(cf, delimiter="\t", lineterminator="\n"), csv.writer(mf, delimiter="\t", lineterminator="\n")
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        def write_found(batch, found) -> None:
            nonlocal total
            for row in batch:
                s1id = row["entity_id"]
                candidates = found.get(s1id, [])
                cw.writerow([s1id, ",".join(entity_id for entity_id, _ in candidates)])
                mw.writerow([s1id, ",".join(entity_id for entity_id, score in candidates if score >= threshold)])
                total += 1
                if s1id in truth:
                    scores.append((candidates, truth[s1id]))
            if total % 10000 < batch_size:
                print(f"Processed {total:,} Source 1 rows", flush=True)

        def batches():
            source1_batch: list[dict[str, str]] = []
            query_rows = read_tsv(s1_path)
            if split == "train" and holdout_percent:
                query_rows = (row for row in query_rows if stable_holdout(row["entity_id"], holdout_percent))
            for row in query_rows:
                source1_batch.append(row)
                if len(source1_batch) >= batch_size:
                    yield source1_batch
                    source1_batch = []
            if source1_batch:
                yield source1_batch

        if workers <= 1:
            for batch in batches():
                write_found(batch, retrieve_batch(db, batch, max_per_source, max_df))
        else:
            db.close()
            pending = set()
            batch_iter = iter(batches())
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, initializer=initialize_worker, initargs=(str(db_path),)
            ) as pool:
                for _ in range(workers * 2):
                    batch = next(batch_iter, None)
                    if batch is None:
                        break
                    pending.add(pool.submit(worker_retrieve, (batch, max_per_source, max_df)))
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        batch, found = future.result()
                        write_found(batch, found)
                        next_batch = next(batch_iter, None)
                        if next_batch is not None:
                            pending.add(pool.submit(worker_retrieve, (next_batch, max_per_source, max_df)))
    db.close()
    if scores:
        print(f"Validation rows scored: {len(scores):,}")
        thresholds = sorted({0.0, threshold, *(round(x / 100, 2) for x in range(20, 101, 2))})
        metrics = []
        for cutoff in thresholds:
            mean = sum(f05({entity_id for entity_id, score in candidates if score >= cutoff}, actual)
                       for candidates, actual in scores) / len(scores)
            metrics.append((mean, cutoff))
        best_score, best_threshold = max(metrics)
        print(f"Macro F0.5 at requested threshold {threshold:.3f}: "
              f"{next(score for score, cutoff in metrics if cutoff == threshold):.6f}")
        print(f"Best sweep threshold: {best_threshold:.2f} (macro F0.5 {best_score:.6f})")
        print(f"Ground-truth singleton rate: {sum(not actual for _, actual in scores)/len(scores):.2%}")
        total_true = sum(len(actual) for _, actual in scores)
        recovered = sum(len({entity_id for entity_id, _ in candidates} & actual)
                        for candidates, actual in scores)
        per_entity_recall = [
            (len({entity_id for entity_id, _ in candidates} & actual) / len(actual))
            for candidates, actual in scores if actual
        ]
        print(f"Blocking link recall: {recovered / total_true:.2%} ({recovered:,}/{total_true:,})" if total_true else "Blocking link recall: n/a (no true links in holdout)")
        print(f"Mean per-entity blocking recall (matched entities): {sum(per_entity_recall)/len(per_entity_recall):.2%}" if per_entity_recall else "Mean per-entity blocking recall: n/a")
    print(f"Wrote {total:,} rows to {matches_path} and {candidates_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"), help="Contains train/ and test/ folders")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--work-db", type=Path, default=Path(".work/entity_index.sqlite"))
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max-candidates-per-source", type=int, default=400)
    parser.add_argument("--max-token-frequency", type=int, default=1000)
    parser.add_argument("--holdout-percent", type=int, default=0, help="Train split only; deterministic S1 holdout percentage")
    parser.add_argument("--reuse-db", action="store_true", help="Reuse a previously built index instead of rebuilding it")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of S1 rows to process per SQL batch")
    parser.add_argument("--workers", type=int, default=12, help="Parallel database readers for candidate generation")
    args = parser.parse_args()
    if args.split == "test" and args.holdout_percent:
        parser.error("--holdout-percent can only be used with --split train")
    run(args.data_dir, args.split, args.output_dir, args.work_db, args.threshold,
        args.max_candidates_per_source, args.max_token_frequency, args.holdout_percent,
        args.reuse_db, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
