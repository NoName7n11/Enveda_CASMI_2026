# CASMI 2026 — Project Log

Dated, reasoned record of what was built and why. Doubles as raw material
for the competition's required winner methodology writeup (Rule 2.5.b)
and as context for future sessions.

## 2026-09-22

### Baseline retrieval pipeline (sub-project 1)

**What:** Built `src/metric.py` (offline MRR@25 scorer using RDKit
tautomer-canonicalized InChIKey14 matching), `src/data.py` (stratified
sampled train-set loader always including the `enveda-np-examples` and
`drug_plus` libraries in full, plus `make_validation_split` for a
molecule-level held-out validation split), and `src/baseline.py`
(precursor-mass + adduct filtering, matchms cosine spectral similarity
scoring, max-aggregation across a molecule's multiple spectra, top-25
submission formatting). Wired into `Enveda_CASMI_local.ipynb`, produced
a first real `submission.csv` against the actual test set.

**Why:** Establishes a working, offline-validatable baseline before any
ML model — per WORKFLOW.md's phased approach (Phase 3). Precursor mass +
cosine similarity is the standard first-pass strategy for Class 1
(publicly-known-spectra) molecules per the competition's own strategic
blueprint (OVERVIEW.md Section 10). Getting a real submission shipped
early also banks MRR@25 points directly (the metric rewards any correct
guess in the top 25, so even a weak baseline contributes to the final
score — unlike a discardable exploratory step).

**Bug found and fixed during final review:** the first version of the
offline validation split held out candidates by `inchikey14` — the exact
match target — meaning the correct structure could never appear in its
own candidate pool. This made the offline MRR@25 number (0.0050)
structurally uninformative rather than just "low" — it measured the
rate of rare tautomer-canonicalization collisions, not retrieval
quality. Fixed by splitting at the spectrum level instead: for each
held-out molecule, some of its spectra become the query set while at
least one other spectrum of that same structure remains in the
candidate pool (mirroring the real Class 1 scenario — the molecule does
have library spectra, just not the exact query acquisition). After the
fix, offline MRR@25 jumped to 0.4173 on the same 75k-row sample,
confirming the harness now actually measures something.

**Result:** 27 tests passing. `src/data.py::make_validation_split` is
the reusable spectrum-level split function later sub-projects (including
this reranker) build on.

## 2026-09-23

### Candidate feature extraction (`src/reranker.py::extract_candidate_features`)

**What:** New function that re-runs the same precursor-mass/adduct
filtering and cosine-similarity scoring as `src/baseline.py`, but
returns a richer per-candidate row: cosine score, precursor ppm error,
candidate peak count, and RDKit descriptors (molecular weight, LogP,
ring count, rotatable bond count, H-bond donor/acceptor counts) computed
from the candidate's SMILES.

**Why:** `src/baseline.py::score_candidates_for_molecule` only returns
`inchikey14`, `smiles`, `score` — not enough signal for a reranker to
learn from. Rather than extending that function's return shape (which
would touch already-shipped, already-reviewed code and risk regressing
its existing callers/tests), a parallel function duplicates the
filter+cosine loop. The duplication cost is small and isolates risk:
`src/baseline.py` and its test suite are untouched by this sub-project.

### Training example builder + AutoGluon training (`build_training_examples`, `train_reranker`)

**What:** `build_training_examples` reuses `make_validation_split`
(already built for baseline offline validation) to get held-out query
molecules and ground truth, extracts candidate features for each via
`extract_candidate_features`, and labels every candidate row
`is_correct` (1 if it matches the molecule's true structure, else 0).
`train_reranker` wraps `autogluon.tabular.TabularPredictor` as a binary
classifier over these labeled rows.

**Why:** Reusing `make_validation_split` means the reranker's training
data and the baseline's offline-eval split share one definition — no
duplicated split logic, and both sub-projects benefit from any future
fix to that function. Framing this as binary classification
(`is_correct` per candidate row) rather than a learning-to-rank
objective is a deliberate v0 simplification: AutoGluon's tabular
binary-classification path is simpler to set up and debug than its
ranking support, and `predict_proba` gives a usable ranking signal
directly. A different acceptable approach for a small candidate pool
per molecule.

**Bug found and fixed during implementation:** `make_validation_split`'s
`val_ground_truth` dict maps `molecule_id -> normalized_smiles` (per its
own docstring), not `-> inchikey14`. The task brief's reference labeling
code compared `features["inchikey14"] == val_ground_truth[molecule_id]`
— an inchikey14 string can never equal a SMILES string, so every row
would have been silently labeled `is_correct=0`, training the model on
uniformly-wrong (all-negative) labels with no visible error. Fixed by
comparing `features["smiles"]` (the candidate's `normalized_smiles`,
already extracted by `extract_candidate_features`) against
`val_ground_truth[molecule_id]` instead. Verified directly: before the
fix, the synthetic 3-molecule fixture produced `is_correct` summing to
0 across all rows; after the fix, it correctly summed to 3 (one true
match per held-out molecule). This also meant the brief's exact test
fixture for `test_train_reranker_fits_and_returns_predictor` (default
`ppm_tolerance=15.0`, precursor_mz values 100/200/300) produces a
single-class ("always 1") training set once labeling is correct, since
each molecule's candidate pool only ever contains its own structure —
AutoGluon's binary classifier cannot fit on one class. That test was
adjusted to pass a much wider `ppm_tolerance` so each molecule's
candidate pool also picks up the other two (incorrect) structures,
producing the 2-class label distribution AutoGluon requires. This is a
test-construction detail only; `build_training_examples`'s labeling
logic itself is unaffected.

**Known limitation (documented in spec, not fixed here):** training
data and the offline-eval held-out split are currently the same split,
so the "reranked MRR@25" comparison in Task 4 should be read as
somewhat optimistic/in-sample. A fully separate double-holdout is a
deferred future refinement.
