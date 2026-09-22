# CASMI 2026 — Workflow

> **How we work:** Local AI-assisted development → Cloud GPU training → Iterate & Submit

---

## High-Level Flow
> [!CAUTION]
> ## Two-Notebook Rule (ALWAYS FOLLOW)
>
> | Notebook | Purpose | When to Use |
> | :--- | :--- | :--- |
> | **`Enveda_CASMI_local.ipynb`** | ALL experimentation, coding, debugging, testing, EDA | **Always first. Every idea starts and is validated here.** |
> | **`Enveda_CASMI_colab.ipynb`** | GPU training ONLY (Google Colab, T4 15 GB) | **Only when code is tested, reviewed, and confirmed ready.** |
>
> **Never write untested or experimental code directly in the Colab notebook.**  
> The local notebook is the workbench. The Colab notebook is the deployment target.  
> Code moves from local → Colab, never the other way around.

---

## Architecture Overview

The workflow is split into **what runs where** based on compute requirements:

```
  LOCAL (You + AI)                 CLOUD (Kaggle / Colab)
  ─────────────────                ────────────────────────
  Plan & strategize          ───►  Train on full dataset
  Write & review code        ───►  Run heavy GPU workloads
  Test on small data         ◄───  Pull results & checkpoints
  Evaluate & iterate         ───►  Submit predictions
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          YOUR LOCAL PC (i7-1260P / 16 GB RAM)                       │
│                                                                                     │
│  ┌─────────────────────────────────┐   ┌──────────────────────────────────────────┐ │
│  │    AI Pair Programmer            │   │    AutoGluon Tabular (CPU)               │ │
│  │    (Antigravity / Claude)        │   │    • Candidate reranking                 │ │
│  │                                  │   │    • LightGBM / CatBoost / XGBoost       │ │
│  │    • Planning & strategy         │   │    • Stacked ensemble training           │ │
│  │    • Code writing & reviews      │   │    • Trained on extracted features        │ │
│  │    • Debugging & optimization    │   │    • Exports portable model artifacts    │ │
│  │    • Metric validation           │   │    • Runs on small-to-medium data slices │ │
│  └─────────────────────────────────┘   └──────────────────────────────────────────┘ │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │    Local Data Work (CPU-only)                                                   │ │
│  │    • EDA on sample subsets (Polars / DuckDB — memory-safe on 16 GB)             │ │
│  │    • RDKit chemistry operations (SMILES ↔ InChIKey14, fingerprints)             │ │
│  │    • matchms spectrum cleaning & cosine similarity                               │ │
│  │    • Feature matrix construction for AutoGluon                                  │ │
│  │    • Offline metric evaluation (official MRR@25 scorer)                          │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬─────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  Push code & pull results via  │
                    │  Kaggle CLI / Google Drive      │
                    └───────────────┬───────────────┘
                                    │
          ┌─────────────────────────┴──────────────────────────┐
          ▼                                                    ▼
┌──────────────────────────────┐          ┌──────────────────────────────────┐
│   KAGGLE NOTEBOOKS (Primary) │          │   GOOGLE COLAB (Fallback)        │
│   • NVIDIA T4 16 GB VRAM     │          │   • NVIDIA T4 15 GB VRAM         │
│   • 30 GB System RAM         │          │   • 12.7 GB System RAM (free)    │
│   • 30 hrs/week free GPU     │          │   • ~12 hrs/day free GPU         │
│   • Dataset pre-loaded (0s)  │          │   • Dataset via Kaggle API (~1m) │
│   • Direct submission button │          │   • Saves to Google Drive        │
│                              │          │                                  │
│   USE FOR:                   │          │   USE FOR:                       │
│   • DL encoder training      │          │   • Overflow DL training         │
│   • Full dataset processing  │          │   • Experimentation when Kaggle  │
│   • Final submission runs    │          │     quota is exhausted           │
└──────────────────────────────┘          └──────────────────────────────────┘
```

### Key Principle: **Write Once, Run Anywhere**
All code is written locally with AI assistance. A single `config.py` auto-detects the environment, so the same `.py` / `.ipynb` file runs identically on Kaggle, Colab, or your local machine.

---

## Phases

### Phase 1 — Research & Understanding
- Study the competition problem, data schema, and evaluation metric.
- Read discussion forums and public notebooks for community insights.
- Identify key challenges (3 novelty classes, multi-spectrum aggregation, adduct handling).

### Phase 2 — Data Exploration
- Inspect training libraries, column distributions, and data quality locally on small subsets.
- Profile the `enveda-np-examples` library (closest match to test instrument).
- Understand collision energy conventions across different source libraries.

### Phase 3 — Baseline Development
- Build a simple retrieval-based baseline (precursor mass filtering + spectral similarity).
- Implement the official MRR@25 evaluation metric locally for offline validation.
- Establish a cross-validation split strategy (molecule-level, not spectrum-level).
- Submit the baseline to Kaggle to get a leaderboard reference score.

### Phase 4 — Feature Engineering & Candidate Retrieval
- Design features that capture spectrum-to-candidate similarity (mass error, peak matching, molecular descriptors).
- Build a candidate retrieval pipeline (formula-based lookup from chemical databases).
- Construct pairwise feature matrices (query spectrum vs. candidate molecule).

### Phase 5 — AutoGluon Reranker (Local, CPU)
- Train an AutoGluon tabular ensemble locally on extracted features to rank candidate molecules.
- Analyze feature importances and failure cases with AI assistance.
- Iterate on feature design based on what the reranker finds useful.

### Phase 6 — Deep Learning Encoders (Cloud, GPU)
- Train neural spectrum encoders on the full 2.5M spectra dataset using Kaggle or Colab GPUs.
- Precompute spectrum embeddings and bring them back locally.
- Feed embeddings as additional features into the AutoGluon reranker and retrain.

### Phase 7 — Iteration & Refinement
- Analyze per-molecule scores to identify weak spots (which novelty class is underperforming?).
- Experiment with alternative architectures, ensembles, and data augmentation.
- Track experiments and scores systematically.

### Phase 8 — Final Submission
- Package all trained model weights as a private Kaggle Dataset.
- Build a self-contained offline inference notebook on Kaggle.
- Select the best 2 final submissions for official judging.

---

## Where Things Run

| Activity | Platform | Why |
| :--- | :--- | :--- |
| Planning, code writing, code review, debugging | **Local PC + AI** | Fast iteration, AI assistance, no quota limits |
| Data exploration on small subsets | **Local PC** | Quick feedback, no upload/download needed |
| AutoGluon tabular training | **Local PC (CPU)** | CPU-optimized algorithms, no GPU needed |
| Offline metric evaluation (MRR@25) | **Local PC** | Instant validation before submitting |
| Full-dataset feature extraction | **Kaggle Notebook (CPU)** | 30 GB RAM, dataset pre-loaded, unlimited CPU quota |
| Deep learning model training | **Kaggle Notebook (GPU)** | Free T4 16 GB, 30 hrs/week |
| Overflow GPU training | **Google Colab (GPU)** | Free T4 15 GB, use when Kaggle quota exhausted |
| Final submission | **Kaggle Notebook** | Required by competition rules (offline, ≤ 9 hrs) |

---

## Kaggle ↔ Colab Switching

When Kaggle's 30-hour weekly GPU quota runs out:

1. Open the same notebook in Google Colab.
2. Mount Google Drive and download the dataset via Kaggle API (one-time, ~1 minute).
3. Run training — the code is platform-agnostic (a small config block at the top auto-detects the environment and adjusts data/output paths).
4. Checkpoints save to Google Drive.
5. When Kaggle quota resets, upload new checkpoints back to Kaggle and resume there.

All model artifacts (weights, embeddings, feature files) use standard portable formats that work identically on Kaggle, Colab, and local.

---

## Daily Routine

```
  MORNING                          MIDDAY                           EVENING
  ───────                          ──────                           ───────
  Review yesterday's results       Push notebook to cloud           Pull training outputs
  AI analyzes failure cases        Launch GPU training               Evaluate MRR@25 locally
  Plan today's experiment          (runs 4–9 hrs in background)     Retrain AutoGluon if needed
  Write & test code locally        Work on other tasks locally      Submit to leaderboard (max 5/day)
                                                                    Log results & plan tomorrow
```

---

## Compute Budget (Weekly, Free Tier)

| Resource | Quota | Usage |
| :--- | :--- | :--- |
| **Kaggle GPU** | 30 hrs/week | Primary GPU training; reserve ~6 hrs for final submission runs |
| **Kaggle CPU** | Unlimited | Data processing, feature extraction |
| **Google Colab GPU** | ~12 hrs/day (free tier) | Overflow training when Kaggle is exhausted |
| **Local PC CPU** | Unlimited | AutoGluon, feature engineering, evaluation, all development |

---

## Artifact Flow

```
  Local PC                    Google Drive (Hub)                Kaggle Datasets
  ────────                    ──────────────────                ───────────────
  AutoGluon model ──────────► Synced copy ──────────────────►  Private Dataset
  Feature matrices ─────────► Synced copy                      (for offline inference)
                              DL checkpoints (from cloud) ──►  Private Dataset
                              Embeddings (from cloud) ─────►   Private Dataset
```

Google Drive acts as the central hub. Trained artifacts flow from wherever they were produced (local or cloud) into Drive, and from there get uploaded to Kaggle as private Datasets for the final offline submission notebook.

---

## Experiment Tracking (MLflow)

MLflow runs as a lightweight, file-based tracker — no server setup required. It logs every experiment across both notebooks into a single unified history.

### What Gets Tracked

| Category | Examples |
| :--- | :--- |
| **Parameters** | Learning rate, batch size, model architecture, feature set used, data subset |
| **Metrics** | MRR@25 (CV), training loss, validation loss, per-class accuracy |
| **Artifacts** | Model checkpoints, submission CSVs, feature importance plots, confusion matrices |
| **Tags** | Notebook source (local / colab), experiment phase, approach name |

### How It Works Across Notebooks

```
  Enveda_CASMI_local.ipynb                  Enveda_CASMI_colab.ipynb
  ────────────────────────                  ────────────────────────
  mlflow tracking URI:                      mlflow tracking URI:
  ./mlruns (local folder)                   /content/drive/MyDrive/casmi2026/mlruns
       │                                              │
       │         (Google Drive sync)                   │
       └──────────────────┬────────────────────────────┘
                          ▼
                   Unified MLflow UI
                   mlflow ui --port 5000
                   (run locally, browse all experiments)
```

- **Local notebook:** MLflow writes to `./mlruns` in the project directory.
- **Colab notebook:** MLflow writes to Google Drive at a synced path. Once Drive syncs locally, all Colab runs appear alongside local runs in the same MLflow UI.
- **Browsing:** Run `mlflow ui` from terminal to view all experiments at `http://localhost:5000`.

### Where MLflow Does NOT Run

> [!NOTE]
> MLflow is **not used** in the final Kaggle submission notebook. That notebook only produces `submission.csv` — no tracking, no logging, no dependencies beyond the model and data.

---

## Key Principles

1. **Write once, run anywhere.** All code is written locally with AI. A config block auto-detects the platform.
2. **CPU work stays local.** AutoGluon, feature engineering, and evaluation never need a cloud GPU.
3. **GPU work goes to cloud.** Deep learning training uses free Kaggle/Colab GPUs.
4. **Seamless failover.** Kaggle and Colab are interchangeable — same code, same formats, same outputs.
5. **AI reviews everything.** Every experiment result, failure case, and design decision is analyzed locally with AI before the next iteration.
6. **Track everything with MLflow.** Every training run, every hyperparameter change, every score — logged and comparable. No more "which settings gave that good score?"
