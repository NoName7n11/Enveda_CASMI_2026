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
so the "reranked MRR@25" comparison in the Task 5 notebook (not the
Task 4 integration test, which independently re-splits with a
different `random_state=2` — see that entry below) should be read as
somewhat optimistic/in-sample. A fully separate double-holdout is a
deferred future refinement.

### Reranking at inference + real-data integration test (`rerank_candidates`)

**What:** `rerank_candidates` takes a trained predictor and a molecule's
candidate features, reorders by predicted `is_correct=1` probability
(explicitly selecting that column rather than assuming position 1 in
`predict_proba`'s output, since AutoGluon does not guarantee column
order), and returns up to `top_k` SMILES. Added
`tests/test_reranker_integration.py`, mirroring the baseline pipeline's
Task 4 pattern: trains and reranks against a real (small) sample of
`dataset/train.parquet` to catch real-data surprises before touching
the notebook.

**Why:** The predict_proba column-order guard is not optional —
silently reading the wrong column would invert every ranking without
raising any error, the worst kind of bug for a scoring pipeline. The
real-data integration test exists for the same reason Task 4 of the
baseline plan added one: synthetic unit-test fixtures don't exercise
real schema quirks (dtypes, missing columns, AutoGluon's own
data-validation behavior) that only show up against the actual parquet
files.

**Verified column selection directly (not just trusted the guard
logic):** printed `predictor.predict_proba(...).columns` against a
real trained predictor and confirmed it returns integer columns
`[0, 1]` (matching `predictor.class_labels == [0, 1]`), so
`probabilities[1]` genuinely selects the positive (`is_correct=1`)
class rather than accidentally reading position 1 by luck.

**Bug found and fixed in the brief's own Step 1 unit-test code:** same
root cause as Task 3's ppm-fixture issue (see Task 3 entry above for
the full explanation) — the brief's three new `test_rerank_candidates_*`
tests used the default `ppm_tolerance=15.0` against
`_make_synthetic_train_df()`, producing a single-class training set
AutoGluon's binary classifier cannot fit. Fixed the same way: pass
`ppm_tolerance=1_000_000.0` to `build_training_examples` in all three
new tests. Test-fixture-construction detail only; `rerank_candidates`'s
own logic required no changes beyond what the brief specified verbatim.

**Real-data integration test result:** ran against a 5,000-row sample
of `dataset/train.parquet` (`n_held_out=20` for both training and the
independent eval split). Completed in ~85s. Printed
`Baseline MRR@25: 0.8000, Reranked MRR@25: 0.8000` — both scores
valid (0.0-1.0) and the pipeline ran end to end without error. The two
scores landing exactly equal here is plausible at this sample size (20
held-out molecules, many likely with small or single-candidate pools
where reordering can't change the top-1 pick) and is consistent with
the brief's smoke-test framing — this test asserts pipeline validity,
not a score improvement threshold.

### Wired into notebook: baseline vs. reranked MRR@25 comparison

**What:** Added cells to `Enveda_CASMI_local.ipynb` training the
reranker on the sampled train set and printing baseline-only MRR@25
alongside reranked MRR@25 on the same held-out split, with the delta.
Model artifact saved to `./autogluon_reranker_model` (gitignored, not
committed — matches the existing `models/`/`checkpoints/` gitignore
pattern intent).

**Why:** This is sub-project 2's actual deliverable per the spec's
success criterion — a direct, printed comparison proving (or
disproving) that reranking helps. No reranked `submission.csv` variant
was produced; that's an explicitly deferred follow-up once the
comparison itself is validated as a real improvement.

**Environment note:** notebook execution repeatedly failed with
`ModuleNotFoundError: No module named 'autogluon'` despite
`autogluon.tabular` being installed in `.venv-reranker`. Root cause:
the `python3` kernelspec's `kernel.json` had a stale/incorrect
`argv[0]` (in one case a bare `"python"` resolving via `PATH` to the
system Python 3.14; in a later fix wave, an absolute path missing the
worktree segment, pointing at a separate `.venv-reranker` copy outside
this worktree). Fixed each time by correcting `kernel.json`'s `argv[0]`
to the worktree-local `.venv-reranker\Scripts\python.exe` absolute
path. Local Jupyter kernel registration issue only — no repo code
changed to fix it.

### Final-review fix wave: labeling metric mismatch, missing gradient-boosted models, overstated result wording

**What:** Fixed four Important + four Minor findings from the final
whole-branch review. (1) `build_training_examples`'s labeling now
compares `features["inchikey14"] == molecule_id` instead of exact
SMILES string equality, matching the competition's actual scoring
criterion (`mrr_at_25` matches by tautomer-canonicalized InChIKey14,
not exact SMILES text — see `src/reranker.py` for the full comment).
(2) Installed the missing optional AutoGluon dependencies
(`autogluon.tabular[lightgbm,catboost,xgboost]==1.6.3`) so gradient-
boosted models actually train instead of silently falling back to
RandomForest/ExtraTrees only. (3) Added two new `tests/test_reranker.py`
tests using a stub predictor with a fixed `predict_proba` output to
assert `rerank_candidates` produces the exact correct order (not just
"a valid permutation"), including a reversed-column-order stub proving
selection is by column name, not position. (4) `train_reranker` now
passes `eval_metric="roc_auc"` to `TabularPredictor` (given the ~4.8%
positive rate, default accuracy-based model selection was nearly
meaningless). Also strengthened the existing labeling test to assert
exactly one positive per molecule, and added `matchms.set_matchms_logger_level("ERROR")`
to the Task 5 setup cell to suppress repeated "No precursor_mz found"
warning spam in cells 8-9's output (cells 1-5, from Task 1-3, are
unchanged and still show the spam — out of scope for this fix wave).

**Result — model types that trained:** after installing the optional
dependencies and re-running the notebook, the AutoGluon leaderboard
shows LightGBMXT, LightGBM, RandomForestGini, RandomForestEntr,
CatBoost, ExtraTreesGini, ExtraTreesEntr, XGBoost, LightGBMLarge, and
WeightedEnsemble_L2 all trained successfully (validation ROC-AUC
0.989–0.997). Only NeuralNetFastAI and NeuralNetTorch still failed
with `ImportError` (fastai/torch were not in this fix wave's scope —
only lightgbm/catboost/xgboost were requested). Best model:
`WeightedEnsemble_L2`, with `Ensemble Weights: {'XGBoost': 1.0}` (i.e.
the ensemble selected pure XGBoost as its best single model this run).

**Result — corrected MRR@25 numbers:** on the same full 75,000-row
sample (`n_held_out=200`, `random_state=42`), with the corrected
InChIKey14-based labeling and the newly-available gradient-boosted
models, `build_training_examples` produced 1,882 training rows (91
positive — one more than the previous run's 90, since the labeling fix
changes which candidate rows are counted correct). The printed
side-by-side comparison on the held-out validation split:

```
Baseline-only MRR@25:  0.4173
Reranked MRR@25:       0.4512
Delta:                 +0.0339
```

**Honest interpretation of this result (correcting prior overstated
wording):** the reranked MRR@25 is higher than baseline on this run,
but training data and the offline-eval split share the same rows (full
train/eval row overlap per `make_validation_split` — a known,
already-documented limitation), and the models that actually trained
(RandomForest/ExtraTrees, and now also LightGBM/CatBoost/XGBoost/their
ensemble) can memorize training rows closely at this data scale. So
this delta is **not** solid evidence of true generalization — it
should be read as a smoke-test signal that the reranking mechanism
works end-to-end, not as a validated improvement. A fully independent
double-holdout (train the reranker on one split, evaluate offline
MRR@25 on a disjoint split) is required before this comparison could
support a generalization claim, and remains a deferred future
refinement per the design spec.

## 2026-09-24

### DreaMS embeddings spike (deferred, no code changes)

**What:** Investigated feasibility of adding DreaMS (pretrained MS/MS
spectrum transformer, https://github.com/pluskal-lab/DreaMS) embeddings
as extra reranker features. Confirmed: MIT licensed, checkpoint publicly
downloadable via Hugging Face Hub, all dependencies (torch==2.2.1,
pyopenms==3.4.0, igraph, wandb, umap-learn, ~20 pinned packages total)
install cleanly in a dedicated Python 3.11 venv — a THIRD isolated Python
environment for this project (system 3.14, `.venv-reranker` 3.12 for
AutoGluon, and this untracked throwaway 3.11 venv for DreaMS). Package
import itself is fast (~26s cold). Embedding API
(`dreams.api.dreams_embeddings`) takes spectra as MGF files, returns
1024-dim vectors.

**Blocker found:** the default embedding model requires downloading
TWO separate large checkpoints (`embedding_model.ckpt`, ~1.18GB,
confirmed downloaded successfully in 160s once retried in isolation; and
a second `ssl_model.ckpt` of unknown size, download in progress when the
spike was stopped) via Hugging Face Hub. Network conditions on the spike
night were unusually slow/inconsistent for large-file HTTP downloads
specifically (small packages installed at normal speed 4-8MB/s; large
wheels like pyopenms and the checkpoints repeatedly stalled for many
minutes before completing or needing a manual kill+retry). This was
network flakiness, not a DreaMS code or environment defect — every
"stall" resolved into real, if slow, progress when finally checked.

**Decision:** deferred. Not worth continuing on this attempt given time
already spent (~2.5 hours), but the path is understood and not
structurally broken: (1) get both checkpoints downloaded in one
uninterrupted session, ideally on better network or pre-fetched via
browser/download manager rather than pip/hf_hub's slow-to-recover
transfer, (2) no CPU throughput number was ever obtained — that
measurement still needs to happen once checkpoints are in place, (3)
consider running on Kaggle/Colab instead of local Windows, since that
environment is pre-built, likely has faster/more reliable bandwidth to
Hugging Face, and matches the cloud-GPU intent WORKFLOW.md already
describes for this phase.

**Cleanup:** all spike artifacts (Python 3.11 venv, cloned DreaMS repo,
downloaded checkpoint, sample MGF file) were throwaway and removed after
the spike — nothing was committed. Python 3.11 itself (installed via
winget) was left on the machine since a future DreaMS attempt will need
it again.

### Reranker double-holdout fix (bounded change)

**What:** `Enveda_CASMI_local.ipynb` previously scored the baseline-vs-
reranked MRR@25 comparison on the SAME `make_validation_split` output
(`random_state=42`) used to train the reranker — an in-sample comparison
per the earlier flagged limitation. Added a new cell drawing a fully
independent held-out split (`make_validation_split(train_df,
n_held_out=200, random_state=99)`) used only for the comparison, never
seen during reranker training. No changes to `src/reranker.py` or
`src/data.py` — `make_validation_split` and `build_training_examples`
already supported this via existing parameters.

**Why:** the previously reported +0.0339 MRR@25 delta was explicitly
caveated as "not solid evidence of generalization" (per the final review
during the reranker sub-project). This fix removes that caveat by
construction — the reported delta now reflects genuine held-out
performance, not memorization by the RandomForest/ExtraTrees/XGBoost
ensemble on rows it trained on.

**Result (real, independent double-holdout):** baseline MRR@25 = 0.3849,
reranked MRR@25 = 0.3946, delta = **+0.0097** (~2.5% relative
improvement). Smaller than the earlier in-sample +0.0339, as expected —
that number was inflated by row overlap. This +0.0097 is the trustworthy
number: the reranker provides a real, if modest, improvement over raw
cosine-similarity ranking on spectra/molecules it never trained on.
(Note baseline MRR@25 also differs slightly from the training split's
0.4173 — 0.3849 here — because it's evaluated on a different random
200-molecule sample, not because of any code change to the baseline
itself.)

### Reranked real submission (bounded change)

**What:** added a new cell to `Enveda_CASMI_local.ipynb` that applies the
trained reranker to the actual competition test set (`test_df`, 400
molecules), reordering each molecule's candidates by predicted
`is_correct` probability instead of raw cosine similarity, and writes
the result to `submission_reranked.csv` (gitignored, not committed) —
separate from the existing baseline-only `submission.csv` so both are
available to compare before choosing which to submit.

**Why:** the notebook already trained the reranker and validated it
offline, but never produced an actual reranked submission file — the
artifact that would actually get uploaded to Kaggle. Kept as a separate
file rather than overwriting `submission.csv` so the baseline version
stays available as a fallback/comparison point.

**Result:** `submission_reranked.csv` written successfully, 400 rows,
format-validated (unique molecule_id, no NaN smiles, ≤25 candidates per
row). Took noticeably longer to generate than the offline validation
runs (~40 minutes vs. a few minutes) because it scores every test
molecule's candidates against the full 75k-row `train_df` pool (not a
held-out validation subset) — this reflects the already-known quadratic
`.iterrows()` cost in `extract_candidate_features`/`score_candidates_
for_molecule` (flagged as a deferred optimization during the baseline
and reranker sub-project reviews), now visibly showing up at full
production scale. Worth optimizing before scaling further (e.g. to the
competition's full ~2.5M-spectra train.parquet on Kaggle).

### Fixed the quadratic candidate-scoring cost (bounded change)

**What:** in both `src/baseline.py::score_candidates_for_molecule` and
`src/reranker.py::extract_candidate_features`, added caching so each
candidate's `matchms.Spectrum` object (and, in the reranker path, its
RDKit descriptors) is built at most once per molecule instead of once
per (test spectrum, candidate) pair. Candidates are re-filtered per test
spectrum since the ppm/adduct window depends on the test row, but the
same `train_df` row (and the same SMILES) frequently reappears across a
molecule's multiple spectra — that repeated, redundant object
construction was the actual bottleneck, not the cosine scoring itself.
Purely additive caching, no change to filtering, scoring, or
aggregation logic.

**Why:** the previous entry's ~40-minute reranked-submission run made
this cost visible at production scale. Rather than a bigger rewrite
(e.g. matchms's batch `CosineGreedy.matrix()` API), the caching fix is
low-risk (behavior-preserving, existing test suite as the safety net)
and targets the actual redundant work directly.

**Result:** timed `build_submission` on a 20-molecule subset of
`test_df` against the full 75k-row `train_df`: 3.57s total (0.179s/
molecule), extrapolating to ~1.2 minutes for the full 400-molecule test
set — down from the ~40 minutes observed before this fix. All 41 tests
still pass unchanged (27 under system Python 3.14 for `src/baseline.py`,
41 under `.venv-reranker`'s Python 3.12 for the full suite including
`src/reranker.py`), confirming the optimization didn't alter any scoring
behavior. This scale of improvement matters directly for eventually
running against the competition's full ~2.5M-spectra `train.parquet`.

### Scaled sample_size 75k → 300k, re-ran full pipeline (bounded change)

**What:** with candidate scoring now ~35x faster, bumped
`load_sampled_train(sample_size=...)` from 75,000 to 300,000 rows in
the notebook (single parameter change, no other code touched) and
re-executed the full pipeline end to end: offline baseline validation,
reranker training, independent double-holdout comparison, and both
real submission files (baseline-only and reranked).

**Why:** more unique training structures should let the reranker learn
a more general is_correct/is_incorrect boundary, and the earlier 75k
result was itself artificially constrained by how slow scoring was
before the caching fix — this was the natural next lever to pull.

**Result:** at 300k rows (144,532 unique structures, up from 75k's
~52,730), reranker training set grew from 1,882 rows/90 positive to
**5,963 rows/97 positive**. Independent double-holdout comparison:
baseline MRR@25 = 0.3883, reranked MRR@25 = **0.4335**, delta =
**+0.0453** (~11.7% relative improvement) — a substantially larger,
still-genuine (disjoint-split) improvement than the 75k run's +0.0097
(~2.5%). More training data measurably helped the reranker generalize
better, not just fit more closely. Both `submission.csv` and
`submission_reranked.csv` regenerated successfully (400 rows each,
format-validated). All 41 tests still passing. Total notebook run time
was longer than 75k scale (network/computation naturally scales with
4x the candidate pool per query), but nowhere near prohibitive — this
result came from a single overnight-adjacent run, not a multi-day
wait, confirming scaling further (e.g. toward the full ~2.5M-spectra
dataset) remains a viable next step rather than a dead end.

### Scaled sample_size 300k → 1M (bounded change)

**What:** same single-parameter change as the 300k step, bumped
`sample_size` to 1,000,000 and re-ran the full pipeline. Monitored
system memory closely before starting (this machine has 16GB RAM, and
Task 4 of the baseline sub-project previously hit a real MemoryError
at large scale before the pyarrow-columnar fix) — free RAM dropped to
~1.1GB at the load's peak, uncomfortably low, but the process completed
without OOM or crash (peaked around 6.6GB working set, well within the
already-fixed columnar-loading path's actual footprint).

**Result:** 242,061 unique structures (up from 300k's 144,532),
reranker training set grew to **16,468 rows/153 positive** (up from
5,963/97). Independent double-holdout: baseline MRR@25 = 0.4732,
reranked MRR@25 = **0.6416**, delta = **+0.1684** (~35.6% relative
improvement) — a much larger jump than the 75k→300k step's own
improvement, more than the 3.3x row-count scale factor alone would
obviously predict. This is a genuine double-holdout result (not the
earlier in-sample bug), but the size of the jump is flagged here for
honesty rather than taken entirely at face value: it may partly reflect
the specific 200 molecules randomly drawn for this run's `random_state=99`
eval split happening to be easier/more represented in the larger 1M
candidate pool, not purely a smooth scaling law. A future run with a
different eval `random_state` would help confirm how much of this is
signal versus this-particular-split variance. Both `submission.csv` and
`submission_reranked.csv` regenerated (400 rows each, format-validated).
All 41 tests still passing. Total wall-clock: ~50 minutes, still well
within a single working session, not yet approaching a scaling wall —
the full ~2.5M-spectra dataset remains a plausible next step, though
this run's tight memory margin (1.1GB free at peak) suggests the full
dataset may need more careful memory management or a machine with more
RAM (e.g. a Kaggle/Colab notebook, which WORKFLOW.md already earmarks
for full-dataset processing) rather than pushing this local machine
further.

### Attempted full 2.5M dataset locally — confirmed impractical (no code changes)

**What:** tried `sample_size=2_600_000` (the full `train.parquet`,
2,539,608 rows) on this local machine to see if the 1M run's tight
memory margin was a one-off or a real ceiling.

**Result:** confirmed a real ceiling. Free system RAM (16GB total)
dropped to 0.1–0.3GB within minutes of the data-loading cell starting,
and the kernel process's CPU usage collapsed from the healthy ~100%
seen in prior runs to ~28%, consistent with heavy OS-level paging/swap
thrashing rather than genuine progress. Killed the process manually
before it either crashed outright or degraded the whole machine's
responsiveness further. No data was written (submission files are only
produced at the very end of the notebook, well past where this failed),
and the notebook file itself was unaffected — `nbconvert` only writes
its output on successful completion, so `Enveda_CASMI_local.ipynb`
still holds the last successful 1M-scale run's results untouched. The
one-line `sample_size` edit was reverted via `git checkout`.

**Decision:** the full 2.5M dataset is not viable on this 16GB local
machine with the current code path — 1M rows already used most of
available RAM, and pyarrow's `Table.take()` approach (the fix from the
baseline sub-project's earlier MemoryError) has a real ceiling well
before the full dataset, not just a slow-but-survivable one. **1M rows
stands as the practical local ceiling** for this project's current
architecture. Scaling further requires either a higher-RAM environment
(Kaggle's 30GB, per WORKFLOW.md's own compute table) or a genuinely
streaming/chunked loading approach instead of loading a full sample
into memory at once — the latter is a larger architectural change, not
attempted here. No code changes were made; this is purely an
operational finding.
