# Enveda CASMI 2026: Molecule ID From Mass Spectra — Comprehensive Overview

> **Competition Link:** [Kaggle Competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)  
> **Host:** Enveda Therapeutics, Inc. (in collaboration with CASMI founders)  
> **Competition Type:** Featured Code Competition (Notebook submission)  
> **Total Prize Pool:** \$50,000 USD  
> **Task:** Predict 2D chemical structures (SMILES) of small molecules from tandem mass spectrometry (LC-MS/MS) spectra.

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Domain Background & Fundamentals](#2-domain-background--fundamentals)
3. [The Challenge & Task Formulation](#3-the-challenge--task-formulation)
4. [Evaluation Metric & Scoring Mechanics](#4-evaluation-metric--scoring-mechanics)
5. [Dataset Architecture & Features](#5-dataset-architecture--features)
6. [Timeline, Phases & Submission Limits](#6-timeline-phases--submission-limits)
7. [Code & Compute Requirements](#7-code--compute-requirements)
8. [Prizes & Awards](#8-prizes--awards)
9. [Advantageous Resources, Tools & References](#9-advantageous-resources-tools--references)
10. [Strategic Modeling Directions & Blueprint](#10-strategic-modeling-directions--blueprint)

---

## 1. Executive Summary

Modern liquid chromatography–tandem mass spectrometry (LC-MS/MS) can detect tens of thousands of chemical species in complex biological extracts (plants, microbes, mammals). However, identifying their exact molecular structures remains a major bottleneck in natural product discovery and metabolomics.

Current standard practice relies heavily on spectral library matching. While effective for known compounds, it fails on the vast majority of unknown molecules that have never been isolated or cataloged. 

In this competition, participants build machine learning models that generate/rank candidate 2D chemical structures represented as SMILES strings for unknown molecules based on their tandem mass spectra. The evaluation revives the legacy of **CASMI** (*Critical Assessment of Small Molecule Identification*), founded in 2012 by Emma Schymanski and Steffen Neumann.

---

## 2. Domain Background & Fundamentals

Mass spectrometers do not measure structures directly; they measure physical properties of ionized molecules and their fragments.

```
+-------------------+      +-------------------------+      +------------------------+      +-------------------------+
| Biological Sample | ---> |       Ionization        | ---> | Precursor m/z Detected | ---> | Collisional Dissociation|
| (Complex Mixture) |      | (M+Adduct)+ or -        |      | (Mass constraint)      |      | (Collision Energy eV)   |
+-------------------+      +-------------------------+      +------------------------+      +-------------------------+
                                                                                                         |
                                                                                                         v
                                                                                           +-------------------------+
                                                                                           |  MS2 Fragment Spectrum  |
                                                                                           |  (m/z vs. Intensities)  |
                                                                                           +-------------------------+
```

### Key Concepts:

1. **Ionization & Adducts:**
   - Instruments detect charged ions. Neutral molecules ($M$) acquire or lose ions, forming adducts.
   - **Positive Ion Mode:** Common adducts include $[M+H]^+$, $[M+NH_4]^+$, $[M+Na]^+$, $[M+K]^+$, $[M-H_2O+H]^+$, $[M-2H_2O+H]^+$.
   - **Negative Ion Mode:** Common adducts include $[M-H]^-$, $[M-H_2O-H]^-$, $[M+CH_2O_2-H]^-$ (formate), $[M+Cl]^-$.
2. **Precursor Mass Detection ($m/z$):**
   - The intact ion's mass-to-charge ratio ($m/z$) is measured with high accuracy (often parts-per-million, ppm). Because charge $z = \pm 1$ for most small molecules, $m/z$ directly constrains the elemental formula and molecular weight.
3. **Fragmentation (MS/MS or MS2):**
   - Precursor ions are collided with an inert gas (e.g., $N_2$ or $Ar$) at specific collision energies.
   - Chemical bonds break in characteristic ways according to bond strengths and thermodynamic stability, producing fragment ions.
4. **Collision Energy:**
   - Expressed in electron volts ($\text{eV}$) or Normalized Collision Energy ($\text{NCE}$). Higher collision energy produces greater fragmentation and smaller fragments. Spectra acquired across different energies give complementary structural clues.
5. **MS2 Spectrum:**
   - A list of detected fragment peaks: $(m/z_i, \text{intensity}_i)$.
   - **Base Peak:** The fragment ion with the highest recorded intensity in the spectrum (normalized to 1.0).
   - **Precursor Peak:** The intact unfragmented precursor ion peak (may or may not be visible depending on energy).

---

## 3. The Challenge & Task Formulation

- **Goal:** Predict the 2D chemical structure (as a SMILES string) for each query molecule.
- **Aggregation:** In the test set, each molecule may have **1 to 16 spectra** (median: 3) recorded under different collision energies or adducts. Predictions are **per `molecule_id`**, NOT per spectrum. You must aggregate information across all spectra of a molecule.
- **Output:** Up to 25 ranked candidate structures per molecule, ordered best-guess first, separated by semicolons:
  ```csv
  molecule_id,smiles
  m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C
  m_004d06,NCCc1ccc(O)cc1;CC(=O)Nc1ccc(O)cc1;OCC(O)CO
  ```

### Test Set Composition & Novelty Classes:
The test set comprises ~1,500 spectra covering ~400 molecules (all acquired on a Bruker timsTOF; monoisotopic mass range 157–1,159 Da, median 348 Da). Molecules fall into 3 novelty tiers (the distribution and molecule assignments are hidden during competition):

| Class | Name | Definition | Expected Strategy |
| :--- | :--- | :--- | :--- |
| **Class 1** | *In public spectral libraries* | Structure has public reference MS/MS spectra in public repositories. | Spectral similarity matching against library spectra. |
| **Class 2** | *Known structure, no public spectra* | No public MS/MS spectra exist, but the structure is in PubChem or COCONUT. | Candidate retrieval / formula calculation + molecular reranking / simulated spectra. |
| **Class 3** | *Novel structure* | Structure is completely absent from PubChem; must be generated *de novo*. | Generative AI / autoregressive SMILES / fragment-based assembly / graph generation. |

---

## 4. Evaluation Metric & Scoring Mechanics

### Metric: Mean Reciprocal Rank @ 25 (MRR@25)

The official evaluation metric is:
$$\text{MRR@25} = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{\text{rank}_u}$$

Where:
- $U$ is the total number of evaluated molecules.
- $\text{rank}_u$ is the position ($1 \le \text{rank}_u \le 25$) of the **first correct guess** for molecule $u$.
- If none of your 25 guesses is correct, the score for that molecule is $0$.
- Each molecule has exactly one ground-truth structure; repeated correct guesses do not add points.
- Scoring scale per molecule:
  - Rank 1: **1.000**
  - Rank 2: **0.500**
  - Rank 3: **0.333**
  - Rank 4: **0.250**
  - Rank 5: **0.200**
  - ...
  - Rank 25: **0.040**
  - Not in top 25: **0.000**

### Structure Matching Rules (How Correctness is Judged):
Ground-truth answers and submitted candidate SMILES are compared using **RDKit (v2026.03.3)**:
1. Parse SMILES into an RDKit Mol object: `Chem.MolFromSmiles(smiles)`
2. Canonicalize tautomers using: `rdMolStandardize.TautomerEnumerator().Canonicalize(mol)`
3. Convert canonicalized molecule to InChIKey: `Chem.MolToInchiKey(canonical_mol)`
4. Extract the **first 14 characters** (`InChIKey14`), which encodes the 2D molecular skeleton (connectivity).
5. Compare the predicted `InChIKey14` to the true `InChIKey14`.

> [!IMPORTANT]
> **Stereochemistry & Tautomers are Ignored!**  
> You are **not** penalized for incorrect stereocenters (e.g., $R/S$, $E/Z$, cis/trans) or tautomeric forms (e.g., keto vs. enol). If the 2D heavy-atom connectivity is identical, it counts as an exact match!
> 
> *Example:* Both `OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O` and `OCC1OC(O)C(O)C(O)C1O` reduce to the same InChIKey14 (`WQZGKKKJIJFFOK`) and score full points.

### Submission File Format Constraints:
- File name: `submission.csv`
- Header: `molecule_id,smiles`
- Up to 25 semicolon-separated candidate SMILES per row.
- Every test `molecule_id` must appear **exactly once**.
- No missing columns, empty files, or `NaN`/null values allowed (triggers `ParticipantVisibleError`).
- No single molecule may have $> 25$ semicolon-separated guesses.
- Invalid/unparseable SMILES strings inside your candidate list simply count as non-matches (they do not crash scoring, but consume a rank).

---

## 5. Dataset Architecture & Features

The dataset comprises **3.04 GB** of compressed Parquet and CSV files.

### 1. `train.parquet` (~2.5 million MS/MS spectra, ~275k unique structures)
Includes 18 columns:
- `molecule_id`: Anonymous identifier for the molecule.
- `spectrum_id`: Unique identifier for each individual acquisition.
- `ms2_mzs`: Array of fragment $m/z$ values.
- `ms2_normalized_intensities`: Aligned array of fragment intensities, normalized such that base peak $= 1.0$.
- `base_peak_intensity`: Raw count of the largest peak before normalization (higher often means higher SNR).
- `adduct`: Standardized precursor adduct.
- `ionization_mode`: `positive` or `negative`.
- `instrument_type`: Machine type (e.g. `timsTOF`, `Orbitrap`, etc.; free-text).
- `precursor_mz`: Experimentally measured precursor $m/z$.
- `collision_energy_ev`: List of converted collision energies in $\text{eV}$ (e.g., `[20]`, `[20, 40, 60]`). **Recommended feature for models.**
- `collision_energy_orig`: Raw collision energy recorded by source (e.g., `40`, `35HCD`, `6V`).
- `collision_energy_orig_units`: Units of original energy (`eV`, `NCE`, `V`, `unknown`).
- `normalized_smiles`: Ground-truth structure standardized with RDKit.
- `inchikey`: Complete InChIKey.
- `inchikey14`: First 14 characters of InChIKey (the 2D skeleton evaluation target).
- `molecular_formula`: Neutral molecular formula.
- `ingest_lib`: Name of the source reference library.
- `adduct_orig`: Unstandardized original adduct text.
- `precursor_error_ppm`: Difference (in ppm) between measured precursor $m/z$ and theoretical adduct mass.
- `num_peaks`: Total number of MS2 fragment peaks.

### 2. Training Libraries (`ingest_lib`) Breakdown

| Library | Spectra | Unique Structures | Notes |
| :--- | :--- | :--- | :--- |
| `enveda-180` | 1,153,785 | 182,941 | Acquired on Bruker timsTOF (identical instrument to test set); synthetic drug-like compounds. |
| `pluskal_ms2` | 527,581 | 46,821 | Orbitrap spectra (MSnLib) of screening/bioactive compounds at multiple NCEs. |
| `riken` | 347,171 | 15,892 | RIKEN public repository, high concentration of plant specialized metabolites. |
| `gnps` | 220,849 | 45,750 | Global Natural Products Social Molecular Networking community libraries. |
| `massbank` | 101,727 | 9,180 | Consortium library aggregating diverse instrument types and labs. |
| `mona` | 92,416 | 11,681 | MassBank of North America (UC Davis Fiehn Lab). |
| `spectraverse` | 50,933 | 9,631 | Harmonized aggregation including niche/obscure public repositories. |
| `msdial` | 40,765 | 9,127 | Public collections distributed with MS-DIAL software. |
| `drug_plus` | 2,545 | 2,539 | Pharmaceutical LC-MS/MS database (~1 spectrum per drug, no collision energy). |
| `enveda-np-examples` | 1,151 | 250 | **Most critical calibration set:** 250 natural products on the exact same timsTOF instrument and pipeline as test data! |
| `masaryk` | 652 | 416 | RECETOX chemical standards (Masaryk University). |

### 3. `test.parquet` (~1,500 spectra, ~400 molecules)
Contains the 12 non-label columns: `molecule_id`, `spectrum_id`, `ms2_mzs`, `ms2_normalized_intensities`, `base_peak_intensity`, `adduct`, `ionization_mode`, `instrument_type` (always `timsTOF`), `precursor_mz`, `collision_energy_ev`, `collision_energy_orig`, `collision_energy_orig_units`.

*Note:* In the local workspace / visible data, `test.parquet` is a placeholder sample sampled from training. During the hidden re-run evaluation, it is replaced with the private test set.

### 4. Ten Standard Test Adducts:
- **Positive:** $[M+H]^+$, $[M+NH_4]^+$, $[M-H_2O+H]^+$, $[M-2H_2O+H]^+$, $[M+Na]^+$, $[M+K]^+$
- **Negative:** $[M-H]^-$, $[M-H_2O-H]^-$, $[M+CH_2O_2-H]^-$, $[M+Cl]^-$

---

## 6. Timeline, Phases & Submission Limits

All deadlines occur at **11:59 PM UTC**:

| Date | Milestone / Event | Description |
| :--- | :--- | :--- |
| **September 14, 2026** | **Competition Start** | Dataset and competition opened. |
| **December 7, 2026** | **Entry Deadline** | Last day to accept competition rules and enter. |
| **December 7, 2026** | **Team Merger Deadline** | Last day to merge teams (max team size: 5). |
| **December 14, 2026** | **Final Submission Deadline** | Competition closes; final submissions evaluated on Private Leaderboard. |

### Operational Limits:
- **Daily Submissions:** Maximum **5 submissions per day**.
- **Final Selection:** You may pick up to **2 final submissions** for official judging.
- **Team Size:** Maximum **5 members**.

---

## 7. Code & Compute Requirements

This is a **Kaggle Code Competition**. Submissions must run through Kaggle Notebooks:
- **Compute Limits:**
  - CPU Notebook runtime: $\le 9\text{ hours}$
  - GPU Notebook runtime: $\le 9\text{ hours}$
- **Internet Access:** **Disabled** during notebook scoring run. All models, packages, and external datasets must be uploaded as Kaggle Datasets beforehand.
- **Output:** Must output a CSV file named `submission.csv` in the current working directory.
- **External Data:** Permitted provided it is freely and publicly accessible to all participants at minimal cost (satisfying the Reasonableness Standard in Rule 2.6). Pre-trained weights and public chemical databases are permitted.
- **Winner Open-Source Requirement:** Winning solutions must be open-sourced under an OSI-approved license (specifically **MIT** per Competition-Specific Terms).

---

## 8. Prizes & Awards

Total Cash Prize Pool: **\$50,000 USD**

- 🥇 **1st Place:** \$16,000
- 🥈 **2nd Place:** \$12,000
- 🥉 **3rd Place:** \$9,000
- 🏅 **4th Place:** \$7,000
- 🎖️ **5th Place:** \$6,000

*Points & Medals:* Kaggle Ranking Points and Competition Medals are awarded according to standard Kaggle rules.

---

## 9. Advantageous Resources, Tools & References

### 1. Official Metric & Baseline Resources
- **Official Metric Notebook:** [`casmi-mean-reciprocal-rank`](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank) — Contains the exact RDKit tautomer canonicalization and `InChIKey14` extraction function.

### 2. Chemical & Spectral Databases
- **PubChem:** >115 million chemical compounds. Essential for candidate retrieval in Class 2.
- **COCONUT (Collection of Open Natural Products):** [coconut.naturalproducts.net](https://coconut.naturalproducts.net/) — Over 400,000 natural product structures with taxonomic provenance.
- **MassBank & MoNA (MassBank of North America):** Curated public tandem MS repositories.
- **GNPS:** Global Natural Products Social Molecular Networking platform.
- **MassIVE:** Repository of billions of raw, unannotated MS/MS spectra for self-supervised pre-training.
- **FragHub:** Harmonized aggregation of open MS/MS reference libraries.

### 3. State-of-the-Art ML Models & Spectrometry Tools
- **DreaMS / GeMS:** 
  - *GeMS:* Curated corpus of millions of unannotated spectra mined from MassIVE.
  - *DreaMS:* Deep learning spectrum-embedding model producing continuous vectors for MS/MS spectra.
- **MIST / MIST-CF:** Neural network models for predicting molecular formulas and fingerprint profiles directly from MS/MS spectra.
- **SIRIUS / CSI:FingerID:** Gold-standard academic software predicting molecular formula (via fragmentation trees) and retrieving structures from structure databases via predicted molecular fingerprints.
- **matchms:** Open-source Python library for MS/MS spectrum curation, filtering, peak cleaning, and cosine similarity calculation.
- **Spec2Vec / MS2DeepScore:** Deep learning embeddings for measuring spectral similarity beyond simple cosine overlap.
- **RDKit:** Essential cheminformatics library for SMILES manipulation, canonicalization, and fingerprint calculation.

---

## 10. Strategic Modeling Directions & Blueprint

To maximize the MRR@25 score across all three novelty classes, top competitors typically deploy a **hybrid tiered architecture**:

```
                              Query MS/MS Spectra
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       Precursor m/z + Adduct                      MS2 Peak List & Intensities
                │                                             │
                ▼                                             ▼
   Formula Prediction (SIRIUS / MIST-CF)          Spectral Embedding (DreaMS / Spec2Vec)
                │                                             │
                ├──────────────────────┬──────────────────────┤
                ▼                      ▼                      ▼
        [CLASS 1 SEARCH]       [CLASS 2 RETRIEVAL]     [CLASS 3 GENERATION]
     Direct Library Matching     Database Filtering      De Novo Generation
       (Train set + GNPS)       (PubChem / COCONUT)     (Autoregressive / VAE)
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       │
                                       ▼
                       Candidate Reranker & Ensembler
                       (MS2 fragmentation tree scoring)
                                       │
                                       ▼
                          Top-25 SMILES (Semicolons)
```

1. **Class 1 (Library Retrieval):**
   - Compute spectral similarity (cosine score, modified cosine, MS2DeepScore) against `train.parquet` and external public libraries (GNPS, MassBank).
   - If a near-perfect spectral match exists with consistent precursor mass, place that structure at Rank 1.
2. **Class 2 (Candidate Search via Formula & Database):**
   - High-accuracy precursor $m/z$ + adduct limits the molecular formula to a tiny handful of candidates.
   - Query PubChem / COCONUT for all structures sharing that exact formula.
   - Rank candidates using MS2 fragmentation predictors (e.g., CFM-ID, MetFrag, or neural fragment predict-and-compare).
3. **Class 3 (De Novo Chemical Structure Generation):**
   - For novel natural products not present in any database, autoregressive sequence models (SMILES transformers, SELFIES models) conditioned on MS/MS embeddings and precursor formula can construct plausible molecular skeletons.
4. **Ensembling & Diversity in Top 25:**
   - Because MRR rewards getting the correct answer as early as possible with no penalty for wrong guesses beyond their rank, allocate Rank 1–5 to high-confidence library/database candidates, and subsequent ranks to diverse structural alternatives.

