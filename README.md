# Business Entity Resolution baseline

This is an initial, reproducible baseline for the Amazon ML Challenge 2026. It uses only the supplied records and Python's standard library. It creates a SQLite inverted index over business name/address tokens, generates a capped candidate list for each Source 1 entity, and scores candidates with transparent token-overlap rules. It does not call external APIs or use outside business data.

This is a **starting baseline**, not a tuned final submission: the score weights and cutoff need to be improved using validation, and the full data run can be I/O intensive. Training data contains no French records, so validation cannot establish performance on France. The open-string country field is used only as a small score feature and is not a country allowlist.

## Requirements

- Python 3.10 or later
- SQLite 3.25 or later (for window functions)
- No third-party Python packages

The code expects the starter archive to have been extracted so that `dataset/train/` and `dataset/test/` exist. From the extracted `student_resource/` directory:

```powershell
python ..\code\business_entity_resolution\src\entity_resolution.py `
  --data-dir dataset --split train --holdout-percent 1 `
  --max-token-frequency 1000 --batch-size 100 --workers 12 `
  --output-dir output\validation --work-db .work\validation.sqlite
```

This holds out a deterministic 1% of Source 1 training entities and prints macro F0.5 for the current cutoff plus a threshold sweep and blocking recall. It writes validation predictions and candidates under `output/validation/`.

Generate test predictions after reviewing validation results:

```powershell
python ..\code\business_entity_resolution\src\entity_resolution.py `
  --data-dir dataset --split test --threshold 0.60 `
  --max-token-frequency 1000 --max-candidates-per-source 400 `
  --batch-size 100 --workers 12 `
  --output-dir ..\..\output --work-db .work\test.sqlite
```

The test run writes to the submission-level `output/` directory. Candidate limits and token-frequency limits can be adjusted with `--max-candidates-per-source` and `--max-token-frequency`. The SQLite work database can become large; keep it on a drive with ample free space. A database stores which split it indexes and cannot be reused for a different split.

## Output format

Both files are UTF-8 TSVs with one row per Source 1 entity. Match/candidate IDs within their respective cells are comma-separated. Empty cells represent no matches/candidates. The final matches are selected from the candidate list, so they are a subset of candidates.

Run the provided validator on `matching_results.tsv` from `student_resource/`:

```powershell
python utils/validate_submission.py `
  --matching ..\..\output\matching_results.tsv `
  --test-dir dataset\test
```

For both files, run the bundled streaming cross-check (it confirms row alignment, duplicate-free lists, ID prefixes, and that final matches are candidates):

```powershell
python ..\code\business_entity_resolution\src\verify_artifacts.py `
  --matching ..\..\output\matching_results.tsv `
  --candidate ..\..\output\candidate_pairs.tsv `
  --expected-rows 1732544
```

The provided validator's optional `--check-ids` mode loads the large Source 2/3 ID sets and can require several GB of memory. The generated IDs come directly from the test Source 2/3 index.

## Current approach and next work

1. Inspect validation errors and improve normalization/blocking while measuring candidate recall.
2. Add richer character and address similarity features, then compare a pairwise model against this heuristic baseline.
3. Tune the final decision threshold for per-entity macro F0.5, with singleton false-match rate reported separately.
4. Re-run the full test pipeline, validate both TSVs, and fill `Documentation_template.md` with measured results.

No GitHub repository is required by the challenge; the reproducible source belongs in the final submission ZIP.
