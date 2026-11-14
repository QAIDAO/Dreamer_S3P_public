from pathlib import Path
import csv
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s3p_repro.common import read_csv, read_json, sha256
from s3p_repro.chemistry import describe_split, screen

ROOT = Path(__file__).resolve().parents[1]

def verify():
    manifest = read_json(ROOT / "data/manifest.json")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Data integrity mismatch: " + item["path"])
    split = describe_split(read_csv(ROOT / "data/classification/train_648.csv"), read_csv(ROOT / "data/classification/test_162.csv"))
    assert split["train"] == {"count": 648, "positive": 324, "negative": 324, "aryl_azo_positive": 322, "aryl_azo_negative": 1}
    assert split["test"] == {"count": 162, "positive": 81, "negative": 81, "aryl_azo_positive": 81, "aryl_azo_negative": 0}
    assert split["shared_murcko_scaffolds"] == split["shared_canonical_structures"] == 0
    selected, audit = screen(read_csv(ROOT / "data/case/generated_20.csv"))
    assert len(audit) == 20 and sum(r["valid"] for r in audit) == 7 and len(selected) == 5
    refs = read_json(ROOT / "data/reference/simulation_three_runs.json")
    for record in refs["per_run"]:
        for method in ["quantum", "classic3d", "classic128"]:
            m = record[method]
            assert m["TP"] + m["FN"] == m["TN"] + m["FP"] == 81
            assert abs((m["TP"] + m["TN"]) / 162 - m["acc"]) < 0.000001
    counts = {}
    for name in ["train", "validation", "test"]:
        with (ROOT / f"data/generation/chebi20/{name}.txt").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert all(r["description"] and r["SMILES"] for r in rows)
        counts[name] = len(rows)
    return {"classification": split, "generation_pairs": counts, "case": {"raw": 20, "valid": 7, "selected": 5}, "reference_confusion_matrices_verified": 9, "files_verified": len(manifest["files"])}

if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
