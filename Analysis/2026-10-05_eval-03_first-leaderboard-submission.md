# Eval 03 — First Leaderboard Submission (2026-10-05)

## What was submitted

- Notebook: `noname7n11/notebooka927747cc9`, Version 2 ("Offline-verified (internet off, self-hosted rdkit wheel)").
- File: `submission.csv` (baseline-only, cosine-similarity retrieval, **no reranker**).
- Offline (local holdout) MRR@25 for this exact file: **0.6740**.
- Why baseline-only and not reranked: Kaggle's "Submit to Competition" dialog had a reproducible bug specifically for `submission_reranked.csv` — `Could not find provided output file submission_reranked.csv`, Submit button stayed disabled, survived a full page reload. `submission.csv` worked fine through the identical flow. User chose to submit the working baseline file now rather than debug further (see prior session's AskUserQuestion decision).

## Result

| | Offline (local) | Public leaderboard |
|---|---|---|
| MRR@25 | 0.6740 | **0.130** |

Submission status: `Succeeded`, scored in ~42 minutes (Kaggle re-ran the full notebook end-to-end under internet-OFF conditions for official scoring, as expected for a Code Competition).

**~5.2x drop from offline to public score.**

## Interpretation

This is the direct empirical confirmation of the methodology gap flagged in [2026-10-04_eval-02_offline-verification-v2.md](2026-10-04_eval-02_offline-verification-v2.md): the offline validation split leaks at least one reference spectrum of each held-out structure into the candidate pool, so the 0.6740/0.83-ish offline numbers only ever measured **Class 1 (library-retrieval-solvable)** performance.

The hidden test set evidently contains a large share of molecules the current pipeline structurally cannot solve:

- **Class 2** (known structure, no public reference spectrum) — nothing to cosine-match against, correct candidate never enters the pool.
- **Class 3** (novel structure, absent from PubChem) — requires de novo generation, not retrieval, at all.

A 0.130 public score is roughly consistent with "the pipeline nails the Class 1 slice it was built for, and scores ~0 on everything else," diluted across the full test set. This matches the predicted failure mode exactly, not a bug — the pipeline is executing correctly; it's solving a narrower problem than the full competition.

## What this changes

- Confirms (not just predicts) that **Class 2 candidate retrieval is the highest-leverage next step** — same reranking infrastructure, different candidate source (PubChem/COCONUT lookup by molecular formula instead of spectral-library matching).
- The "honest validation split" recommendation from eval-02 (remove *all* spectra of a held-out structure from the candidate pool, not just some) is now worth building — it would have predicted this drop locally instead of costing a submission slot to discover it. Still not actioned.
- 4 of 5 daily submission slots remain. The better-performing `submission_reranked.csv` (offline 0.8314) has still not been submitted — expected to also score far below its offline number for the same reason, so submitting it mainly checks whether the reranker helps at all on the real distribution, not whether it closes the Class 2/3 gap (it can't, by construction).

## Status

- First real leaderboard signal obtained. Hypothesis from eval-02 confirmed quantitatively.
- Next decision point (not yet actioned): build Class 2 retrieval vs. submit reranked version to use a slot vs. build honest local validation first.
