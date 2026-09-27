# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Noob Coders
**Team Members:** Isha Mahadev, Arnav Tripathi, Daksh Vyas, Aman Modi
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We implemented an end-to-end Entity Resolution pipeline utilizing a SQLite inverted token index for scalable candidate generation, paired with a finely-tuned XGBoost classifier for precise matching. By integrating Levenshtein string similarities alongside token overlap metrics, we improved our validation F0.5 score from 0.579 to a robust 0.720, completely free of any external data APIs.

---

## 2. Methodology

### 2.1 Problem Analysis
During our initial data analysis, we noticed that commercial entities often share noisy string artifacts like "inc", "corp", or "pvt". Standard string distance fails when word order swaps (e.g. "Pharmacy CVS" vs "CVS Pharmacy"). Relying on strict country matches was also flawed because we anticipated an unseen country (France) in the test data.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier
**Core Innovation:** A lightweight SQLite-based inverted index for memory-efficient initial blocking (using rare tokens), followed by a tree-based ML model (XGBoost) trained on robust sequence metrics (difflib) and Jaccard similarities to catch both typos and transpositions safely.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Inverted token index utilizing term-frequencies (TF). We aggressively filtered out stop words and common tokens using a max document frequency (`max_df`).
- **Candidate pairs generated:** Top 50 candidates per Source 1 entity (fetching up to ~3.75 million pairs for ML model processing).
- **How you ensured true matches were not lost:** We joined on the rarest tokens available in the query string's name or address, minimizing false-negative filtering.

---

## 4. Matching Model

**Features used:**
- Name features: Token Jaccard overlap, string containment ratios, and Levenshtein-based sequence similarity.
- Address features: Address Jaccard overlap, string containment, Levenshtein-based sequence similarity.
- Other: Exact country match binary flag (falling back gracefully to 0 when unseen countries like France appear).

**Model type:** XGBoost Classifier (under 8B parameters, Apache 2.0 License).
**Threshold selection method:** F_0.5 optimization on a 1% holdout validation set, selecting `0.98` as the optimal cutoff to aggressively penalize false merges and preserve singletons.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.720 
- **Common false positives (wrong merges):** Franchises or generic names (e.g., "Main St Cafe" in two different states where address data was sparsely populated).
- **Common false negatives (missed matches):** Entities where the token representations differed completely (e.g., heavy abbreviation vs full un-abbreviated text without any rare overlapping token).

---

## 6. Conclusion
By pairing an extremely scalable SQL blocking strategy with an XGBoost ML classifier, we achieved high precision without sacrificing speed or breaching memory limits. Our approach remains entirely self-contained, generalizing safely to unseen countries like France while maximizing the strict F0.5 metric.

---

## Appendix

### A. Code Artefacts
Our pipeline is reproducible entirely from the root directory via the following commands:
1. `python src\entity_resolution.py` - Runs the pipeline, extracts candidates, extracts features, queries the XGBoost model, and writes directly to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
2. `src\generate_features.py` & `src\train_model.py` - Auxiliary scripts used for producing the model artifact.

All dependencies (pandas, xgboost) are pinned in `requirements.txt`.
