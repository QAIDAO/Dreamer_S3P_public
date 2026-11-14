from __future__ import annotations
from collections import Counter
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog("rdApp.error")
ARYL_AZO = Chem.MolFromSmarts("[c]-[N]=[N]-[c]")
NITRO = Chem.MolFromSmarts("[N+](=O)[O-]")

def molecule(smiles):
    return Chem.MolFromSmiles(smiles)

def fingerprints(rows, radius=2, bits=128):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    values = []
    for row in rows:
        mol = molecule(row["smiles"])
        if mol is None:
            raise ValueError(f"Invalid classification structure: {row['molecule_id']}")
        values.append(generator.GetFingerprintAsNumPy(mol))
    return np.asarray(values, dtype=float)

def screen(rows, qed_min=0.5, sa_max=3.0):
    accepted, audit = [], []
    for row in rows:
        mol = molecule(row["smiles"])
        item = {"molecule_id": row["molecule_id"], "smiles": row["smiles"], "valid": mol is not None}
        if mol is None:
            item.update({"selected": False, "reason": "invalid_structure"})
        else:
            qed = float(QED.qed(mol))
            sa = float(sascorer.calculateScore(mol))
            selected = qed >= qed_min and sa <= sa_max
            item.update({"qed": qed, "sa": sa, "aryl_azo": bool(mol.HasSubstructMatch(ARYL_AZO)), "nitro": bool(mol.HasSubstructMatch(NITRO)), "selected": selected, "reason": "selected" if selected else "property_threshold"})
            if selected:
                accepted.append({"molecule_id": row["molecule_id"], "smiles": row["smiles"], "qed": qed, "sa": sa})
        audit.append(item)
    return accepted, audit

def describe_split(train, test):
    def summarize(rows):
        scaffolds, structures = set(), set()
        counts, azo = Counter(), Counter()
        for row in rows:
            mol = molecule(row["smiles"])
            if mol is None:
                raise ValueError(f"Invalid labeled structure: {row['molecule_id']}")
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError("Labels must be 0 or 1")
            counts[label] += 1
            azo[label] += int(mol.HasSubstructMatch(ARYL_AZO))
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
            structures.add(Chem.MolToSmiles(mol))
        return {"count": len(rows), "positive": counts[1], "negative": counts[0], "aryl_azo_positive": azo[1], "aryl_azo_negative": azo[0]}, scaffolds, structures
    tr, tr_scaf, tr_smi = summarize(train)
    te, te_scaf, te_smi = summarize(test)
    return {"train": tr, "test": te, "shared_murcko_scaffolds": len(tr_scaf & te_scaf), "shared_canonical_structures": len(tr_smi & te_smi), "scaffold_definition": "RDKit Bemis-Murcko, includeChirality=False; empty scaffold treated as one group"}
