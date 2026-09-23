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
