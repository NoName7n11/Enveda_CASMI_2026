"""Official CASMI 2026 MRR@25 metric: InChIKey14-based structure matching."""

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def canonicalize_to_inchikey14(smiles: str) -> str | None:
    """Parse SMILES, canonicalize tautomers, return the 14-char InChIKey prefix.

    Returns None if the SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_mol = _tautomer_enumerator.Canonicalize(mol)
    inchikey = Chem.MolToInchiKey(canonical_mol)
    if not inchikey:
        return None
    return inchikey[:14]


def reciprocal_rank(predicted_smiles: list[str], true_smiles: str, k: int = 25) -> float:
    """1/rank of the first predicted SMILES matching true_smiles's InChIKey14.

    Unparseable predicted SMILES occupy a rank slot but never match.
    Returns 0.0 if no match is found within the first k predictions.
    """
    true_key = canonicalize_to_inchikey14(true_smiles)
    for rank, smiles in enumerate(predicted_smiles[:k], start=1):
        predicted_key = canonicalize_to_inchikey14(smiles)
        if predicted_key is not None and predicted_key == true_key:
            return 1.0 / rank
    return 0.0


def mrr_at_25(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, str],
    k: int = 25,
) -> float:
    """Mean reciprocal rank @ k over all molecules in ground_truth.

    Raises KeyError if a ground_truth molecule_id is missing from predictions.
    """
    scores = [
        reciprocal_rank(predictions[molecule_id], true_smiles, k=k)
        for molecule_id, true_smiles in ground_truth.items()
    ]
    return sum(scores) / len(scores)
