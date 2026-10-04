# Evaluation 02 — Offline Verification Run, Version 2 (internet OFF) + Deep Logic Audit

**Date:** 2026-10-04
**Notebook:** `noname7n11/notebooka927747cc9`, Version 2 ("Offline-verified (internet off, self-hosted rdkit wheel)")
**Scope:** (1) Verify the pipeline survives the competition's actual internet-OFF scoring condition. (2) Full line-by-line audit of code + live outputs against OVERVIEW.md's spec, requested directly by the user without reloading the page (i.e. auditing the exact state Kaggle had already produced).

---

## Part 1 — Offline (internet OFF) run

### Problem found

Competition requires internet **disabled** during scoring (OVERVIEW.md §7). The notebook's pip-install cell (`!pip install -q rdkit`) needs PyPI access. Toggling internet off and running confirmed real failure:

```
WARNING: Retrying ... Temporary failure in name resolution ... /simple/rdkit/
ERROR: Could not find a version that satisfies the requirement rdkit (from versions: none)
ERROR: No matching distribution found for rdkit
...
ModuleNotFoundError: No module named 'rdkit'
```

rdkit is **not** pre-installed on Kaggle's base image in a way pip can see offline.

### Fix

1. Downloaded the official rdkit wheel directly from PyPI (not a third-party Kaggle dataset, to avoid supply-chain trust risk): `rdkit-2024.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`, confirmed its only declared dependencies (`numpy`, `Pillow`) are already present on Kaggle's image.
2. Uploaded it as a private Kaggle Dataset: `noname7n11/rdkit-2024-3-2-cp312-offline-wheel`.
3. Attached it as notebook Input.
4. **First path guess was wrong** — tried `/kaggle/input/rdkit-2024-3-2-cp312-offline-wheel/` (flat), got `WARNING: Location ... is ignored: it is either a non-existing path or lacks a specific scheme.` A diagnostic `!ls /kaggle/input/` revealed the real top-level structure is `competitions  datasets` (not a flat list of slugs) — mirrors the exact same surprise found earlier for competition data. Real path: `/kaggle/input/datasets/<username>/<slug>/`.
5. Fixed install command: `!pip install -q --no-index --find-links=/kaggle/input/datasets/noname7n11/rdkit-2024-3-2-cp312-offline-wheel rdkit`.

### Result — offline run succeeded, no errors

| Metric | Internet ON (Eval 01) | Internet OFF (this run) |
|---|---|---|
| Baseline-only MRR@25 | 0.6740 | **0.6740** (exact match) |
| Reranked MRR@25 | 0.8393 | **0.8314** |
| Delta | +0.1653 | +0.1574 |

Reranked score differs slightly (0.8393 vs 0.8314) — expected run-to-run variance in `sklearn.ensemble.GradientBoostingClassifier` (not every randomness source is pinned across reinstalls/threads). Baseline is deterministic and matches exactly, as expected (no learned model involved).

`submission.csv` and `submission_reranked.csv` both written, 400 rows each. Committed as official Version 2.

**Conclusion: the notebook is mechanically submittable** — runs clean under the exact internet-OFF condition the competition scoring uses, within the 9-hour limit (~46 min actual).

---

## Part 2 — Deep logic audit (code + outputs vs. competition spec)

Requested by user directly on the live, already-run Kaggle page (not re-run, analyzed as-is). Full line-by-line review of all 6 inlined-module cells plus the pipeline cells, cross-checked against `OVERVIEW.md`.

### Code correctness — confirmed correct

- `canonicalize_to_inchikey14`: matches OVERVIEW §4 exactly — `MolFromSmiles` → `TautomerEnumerator().Canonicalize` → `MolToInchiKey` → `[:14]`.
- `reciprocal_rank`: unparseable SMILES consume a rank slot but never match; returns 0.0 if nothing found in top-k. Matches spec.
- **Aggregation is per `molecule_id`, not per spectrum** (`test_df.groupby("molecule_id")`, max-cosine-score aggregation across a molecule's spectra) — this is the single most common mistake people make on this competition, and it's implemented correctly here.
- Double-holdout is genuine: reranker trains on `random_state=42` split, evaluates on an independent `random_state=99` split. No leakage between the two.
- `rerank_candidates` selects the positive class via `predictor.classes_.index(1)` rather than assuming a fixed column position — correct and defensive.
- Submission format constraints (unique `molecule_id`, no NaN, ≤24 semicolons = ≤25 candidates, every test ID present) are all asserted in-notebook and pass.

### Critical finding — validation methodology measures the wrong task

**`make_validation_split`'s design:** for each held-out molecule, it holds out *some* of that molecule's spectra as queries while deliberately leaving **at least one other spectrum of the same structure** in the candidate pool (`val_train_df`). Verified empirically:

```python
# 50/50 held-out molecules had their TRUE structure still present in the candidate pool
present = sum(1 for k in gt.keys() if k in set(vt['inchikey14']))
# -> 50/50
```

This means the offline score measures: *"given the correct answer's structure is already in the reference library, can we retrieve it?"* — a pure library-lookup task.

**But OVERVIEW.md §3 defines three novelty classes for the real hidden test set:**

| Class | Definition | Does this pipeline handle it? |
|---|---|---|
| Class 1 — in public spectral libraries | Structure has public reference MS/MS spectra | ✅ Yes — this is exactly what's built |
| Class 2 — known structure, no public spectra | Structure is in PubChem/COCONUT but has no reference spectrum | ❌ No — nothing to cosine-match against; candidate not retrievable |
| Class 3 — novel structure | Structure absent from PubChem entirely | ❌ No — requires de novo generation, not retrieval |

The pipeline is **pure library retrieval**: `filter_candidates` only ever returns rows already present in `train_df`. For Class 2 and Class 3 molecules, the correct SMILES is structurally impossible for this pipeline to emit — those molecules score exactly 0 regardless of how good the ranking model is.

Two corroborating facts:
1. OVERVIEW.md §5.3 states local `test.parquet` is explicitly "a placeholder sample sampled from training" — explains why retrieval looks artificially strong locally; the real hidden test set will not have this property for all molecules.
2. Kaggle's own official tutorial notebook for this competition is literally named `casmi-denovo-tutorial-notebook` and uses a transformer/generative approach, not library matching — strong signal the intended solution space goes beyond retrieval.

### Conclusion

**0.8314 (or 0.8393) is an upper bound on Class-1-only performance, not a real estimate of leaderboard score.** If the hidden test set has a meaningful fraction of Class 2/3 molecules (plausible — Enveda chose this 3-tier framing deliberately), true leaderboard MRR@25 will be substantially lower than the offline number suggests.

### Recommendations (not yet actioned as of this writeup)

1. **Submit this version to the real leaderboard anyway.** Costs one of 5 daily submission slots; the gap between offline (0.83) and public leaderboard score directly reveals the real Class 1/2/3 distribution — the single most valuable currently-unknown fact for planning next steps.
2. **Build an honest validation split.** A variant of `make_validation_split` that removes *all* spectra of a held-out structure from the candidate pool (not just some) would approximate Class 2/3 performance (expected to be ~0 for the current pipeline) and make this gap visible locally instead of only on the leaderboard.
3. **Class 2 is the next tractable improvement.** Candidate retrieval from PubChem/COCONUT by molecular formula (rather than by spectral similarity to a library spectrum), reranked with the same feature-engineering/reranker infrastructure already built — only the candidate *source* needs to change, not the reranking machinery.
4. Class 3 (true de novo generation) is a substantially larger undertaking (generative SMILES modeling) and likely out of scope unless Class 2 work is exhausted first.

## Status at end of this evaluation

- Notebook confirmed submittable (passes internet-off requirement, Version 2 committed).
- Real methodology gap identified and documented before submitting to the real leaderboard — avoids wasting a submission slot on pure guesswork about why a leaderboard score might differ from offline.
- Next decision point: submit now to calibrate against real distribution, vs. build Class 2 support first. Not yet decided as of this writeup.
