"""Class 2 candidate retrieval: formula-based search, no spectral match required.

Class 1 (src.baseline) retrieves candidates by spectral cosine similarity --
it can only ever find a structure that already has a reference spectrum in
the pool. Class 2 molecules (OVERVIEW.md Section 3) have no public reference
spectrum at all, so cosine similarity has nothing to compare against; the
correct structure is unreachable for src.baseline by construction (confirmed
empirically: see Analysis/2026-10-04_eval-02_offline-verification-v2.md and
the 0.130 public-score result in Analysis/2026-10-05_eval-03).

OVERVIEW.md Section 6's recommended Class 2 approach is: use the precursor
m/z + adduct to constrain the molecular formula, then query a structure
database (PubChem/COCONUT) for every structure sharing that formula, then
rerank those candidates (no spectral features available -- only
structure-level ones).

This module implements that retrieval + reranking mechanism against any
"structure table" with a `molecular_formula` and `normalized_smiles`/`smiles`
column -- today that's necessarily `train_df` (the only structure table this
repo has offline access to), which makes this only a partial Class 2 fix:
it will find train-set structures sharing the test molecule's formula even
when no reference spectrum of that exact structure exists, but it still
can't reach a structure that's absent from train_df entirely (true
PubChem/COCONUT-only structures). Swapping the candidate source to a
self-hosted offline PubChem/COCONUT formula index (same offline-dataset
trick already used for the rdkit wheel) would close that remaining gap
without changing the retrieval/reranking logic here.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.reranker import _rdkit_descriptors

_FEATURE_COLUMNS = [
    "inchikey14", "smiles", "ppm_error",
    "mol_wt", "log_p", "num_rings",
    "num_rotatable_bonds", "num_hbd", "num_hba",
]


def formula_from_precursor(precursor_mz: float, adduct: str) -> str | None:
    """Best-effort placeholder for deriving a molecular formula from m/z + adduct.

    Real formula calculation from accurate mass requires solving the
    (highly ambiguous for small mass windows) integer-composition problem
    and is out of scope here -- this repo's data already carries
    `molecular_formula` per row (OVERVIEW.md Section 5), so
    filter_candidates_by_formula is normally called with a known formula
    string rather than one derived from mass. This function exists only so
    a caller without a formula column can plug in a real formula predictor
    (e.g. SIRIUS, MIST-CF) later without changing filter_candidates_by_formula's
    interface. Returns None (not implemented).
    """
    return None


def filter_candidates_by_formula(
    molecular_formula: str, structure_table: pd.DataFrame
) -> pd.DataFrame:
    """Return every unique structure in structure_table sharing an exact formula.

    Unlike src.baseline.filter_candidates, this ignores adduct/precursor_mz/
    spectral content entirely -- Class 2 molecules have no reference
    spectrum to match against, so formula is the only retrieval signal
    available. Deduplicates to one row per inchikey14 (a formula can map to
    many spectra of the same structure).
    """
    if molecular_formula is None or (isinstance(molecular_formula, float) and np.isnan(molecular_formula)):
        return structure_table.iloc[0:0]
    matches = structure_table[structure_table["molecular_formula"] == molecular_formula]
    return matches.drop_duplicates("inchikey14")


def extract_structure_features(
    test_row: pd.Series, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Per-candidate structure-only features (no spectral cosine_score available).

    One row per candidate inchikey14. ppm_error compares the candidate's
    monoisotopic mass (computed from its own SMILES, not a reference
    spectrum's precursor_mz, since Class 2 candidates have none) against the
    test molecule's observed precursor m/z.
    """
    if candidates.empty:
        return pd.DataFrame(columns=_FEATURE_COLUMNS)

    rows = []
    descriptor_cache: dict[str, dict] = {}
    for _, candidate_row in candidates.iterrows():
        smiles = candidate_row["normalized_smiles"]
        if smiles not in descriptor_cache:
            descriptor_cache[smiles] = _rdkit_descriptors(smiles)
        mol = Chem.MolFromSmiles(smiles)
        candidate_mass = rdMolDescriptors.CalcExactMolWt(mol) if mol is not None else np.nan
        ppm_error = (
            abs(candidate_mass - test_row["precursor_mz"]) / test_row["precursor_mz"] * 1e6
            if mol is not None
            else np.nan
        )
        rows.append(
            {
                "inchikey14": candidate_row["inchikey14"],
                "smiles": smiles,
                "ppm_error": ppm_error,
                **descriptor_cache[smiles],
            }
        )
    return pd.DataFrame(rows, columns=_FEATURE_COLUMNS)


def build_class2_submission(
    test_df: pd.DataFrame, structure_table: pd.DataFrame, top_k: int = 25
) -> pd.DataFrame:
    """Build a molecule_id -> top-k semicolon-joined SMILES submission via formula retrieval.

    For each test molecule, candidates are every distinct structure in
    structure_table sharing its molecular_formula, ranked by ppm_error
    (closest exact mass first) -- the only ranking signal available without
    a trained reranker or any spectral evidence. Use
    src.reranker.train_reranker on this module's _FEATURE_COLUMNS (minus
    cosine_score, which doesn't exist here) to replace this naive ordering
    with a trained one; the feature-extraction/training infrastructure
    already built for Class 1 (src/reranker.py) transfers directly.
    """
    rows = []
    for molecule_id, group in test_df.groupby("molecule_id"):
        test_row = group.iloc[0]
        candidates = filter_candidates_by_formula(test_row.get("molecular_formula"), structure_table)
        if candidates.empty:
            rows.append({"molecule_id": molecule_id, "smiles": ""})
            continue
        features = extract_structure_features(test_row, candidates)
        top = features.sort_values("ppm_error", ascending=True, na_position="last").head(top_k)
        rows.append({"molecule_id": molecule_id, "smiles": ";".join(top["smiles"])})
    return pd.DataFrame(rows)
