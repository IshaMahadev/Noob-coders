#!/usr/bin/env python3
import csv
import sqlite3
import difflib
import argparse
from pathlib import Path
from entity_resolution import normalize, load_truth, read_tsv, retrieve_batch, candidate_features

def string_sim(a: str, b: str) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("../dataset"))
    parser.add_argument("--db-path", type=Path, default=Path("../.work/validation.sqlite"))
    parser.add_argument("--out-csv", type=Path, default=Path("../training_features.csv"))
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    truth = load_truth(args.data_dir / "train" / "train_ground_truth.tsv")
    s1_path = args.data_dir / "train" / "train_source1.tsv"

    if not args.db_path.exists():
        print("Database not found. Please run entity_resolution.py first to build the index.")
        return

    db = sqlite3.connect(args.db_path)
    db.row_factory = sqlite3.Row

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "s1_id", "cand_id", "source", "label",
            "name_jaccard", "addr_jaccard", "name_contain", "addr_contain",
            "name_seq_sim", "addr_seq_sim", "country_match"
        ])

        print("Generating features using optimized batching...")
        
        batch = []
        total_processed = 0
        
        def process_batch(current_batch):
            # 1. Use the highly optimized batch retrieval from the baseline
            retrieved = retrieve_batch(db, current_batch, max_per_source=50, max_df=1000)
            
            # 2. Fetch the actual record strings for the retrieved candidates
            cand_ids = set()
            for cands in retrieved.values():
                for cid, _ in cands:
                    cand_ids.add(cid)
                    
            if not cand_ids:
                return
                
            cand_ids_list = list(cand_ids)
            records = []
            for i in range(0, len(cand_ids_list), 500):
                chunk = cand_ids_list[i:i+500]
                placeholders = ",".join("?" for _ in chunk)
                records.extend(db.execute(f"SELECT * FROM records WHERE entity_id IN ({placeholders})", tuple(chunk)).fetchall())
            cand_dict = {r["entity_id"]: r for r in records}
            
            # 3. Calculate all ML features and write to CSV
            for row in current_batch:
                s1_id = row["entity_id"]
                a_name, a_addr = normalize(row.get("business_name", "")), normalize(row.get("business_address", ""))
                
                candidates = retrieved.get(s1_id, [])
                for cand_id, _ in candidates:
                    cand = cand_dict[cand_id]
                    label = 1 if s1_id in truth and cand_id in truth[s1_id] else 0
                    
                    nj, aj, nc, ac = candidate_features(row, cand)
                    name_seq = string_sim(a_name, cand["name_norm"])
                    addr_seq = string_sim(a_addr, cand["address_norm"])
                    country_match = 1 if row.get("country") == cand["country"] and row.get("country") else 0

                    writer.writerow([
                        s1_id, cand_id, cand["source"], label,
                        nj, aj, nc, ac, name_seq, addr_seq, country_match
                    ])

        for row in read_tsv(s1_path):
            batch.append(row)
            if len(batch) >= args.batch_size:
                process_batch(batch)
                total_processed += len(batch)
                print(f"Processed {total_processed} rows...")
                batch = []
                
        if batch:
            process_batch(batch)
            total_processed += len(batch)
            print(f"Processed {total_processed} rows...")
                
    print(f"Features saved to {args.out_csv}")

if __name__ == "__main__":
    main()
