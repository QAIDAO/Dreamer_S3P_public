from __future__ import annotations
from pathlib import Path
import importlib.metadata
import json
import time
from .common import read_csv, read_json, write_csv, write_json, sha256, digest
from .chemistry import screen, describe_split
from .quantum import FeatureBackend
from .classification import Classifier

def run(root, output, config, *, backend_name="simulator", seed=42, sample_shots=False, candidate_file=None, generator_model=None, worker_python=None, allow_hardware=False, retry_unknown=False):
    root, output = Path(root), Path(output)
    train_file = root / "data/classification/train_648.csv"
    test_file = root / "data/classification/test_162.csv"
    candidate_file = Path(candidate_file) if candidate_file else root / "data/case/generated_20.csv"
    if generator_model and backend_name != "simulator" and not allow_hardware:
        raise ValueError("Plan and authorize hardware before generating candidates")
    model_hashes = None
    if generator_model:
        directory = Path(generator_model)
        paths = [directory / "config.json", directory / "tokenizer.json", *sorted(directory.glob("*.safetensors"))]
        model_hashes = {p.name: sha256(p) for p in paths if p.is_file()}
    manifest = {"schema_version": 1, "new_execution": True, "historical_replay": False, "execution_source": backend_name, "is_hardware": backend_name == "spinq-nmr", "seed": seed, "simulator_mode": "multinomial_sampling" if sample_shots else "analytic_statevector", "shots": 128, "circuit_layers": 1, "train_sha256": sha256(train_file), "test_sha256": sha256(test_file), "candidate_input_sha256": None if generator_model else sha256(candidate_file), "generator_model_checksums": model_hashes, "config_sha256": digest(config), "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn", "rdkit"]}}
    key = digest(manifest)
    if (output / "manifest.json").exists() and read_json(output / "manifest.json").get("run_sha256") != key:
        raise ValueError("Output belongs to a different run; choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest["run_sha256"] = key
    write_json(output / "manifest.json", manifest)
    def event(stage, status, **extra):
        with (output / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"stage": stage, "status": status, **extra}, ensure_ascii=False, allow_nan=False) + "\n")
    started = time.monotonic()
    try:
        event("inputs", "running")
        train, test = read_csv(train_file), read_csv(test_file)
        write_json(output / "split_audit.json", describe_split(train, test))
        event("inputs", "completed")
        if generator_model:
            from .generation import generate
            raw, _ = generate(generator_model, output / "generation", config)
            event("generation", "completed", source="local_conditional_generator", count=len(raw))
        else:
            raw = read_csv(candidate_file)
            event("generation", "replayed", source="provided_smiles_snapshot", count=len(raw))
        accepted, audit = screen(raw, config["screen"]["qed_min"], config["screen"]["sa_max"])
        write_json(output / "screening.json", audit)
        write_csv(output / "filtered.csv", accepted)
        event("screening", "completed", valid=sum(a["valid"] for a in audit), selected=len(accepted))
        backend = FeatureBackend(output / "feature_cache", name=backend_name, seed=seed, sample_shots=sample_shots, worker_python=worker_python, worker_script=root / "adapters/spinq_worker.py", allow_hardware=allow_hardware, retry_unknown=retry_unknown)
        event("classification", "running")
        classifier = Classifier(backend, seed).fit(train)
        result, predictions = classifier.evaluate(test)
        ranking = classifier.rank(accepted)
        classifier.save_arrays(output / "classifier_arrays.npz")
        write_json(output / "metrics.json", {"record_type": "new_local_evaluation", "execution_source": backend_name, "results": result})
        write_csv(output / "test_predictions.csv", predictions)
        write_csv(output / "ranking.csv", ranking)
        write_json(output / "feature_execution.json", backend.records)
        summary = {"run_sha256": key, "execution_source": backend_name, "is_hardware": backend_name == "spinq-nmr", "new_classification": True, "generation_source": "local_conditional_generator" if generator_model else "provided_smiles_snapshot", "train_samples": len(train), "test_samples": len(test), "raw_strings": len(raw), "valid_structures": sum(a["valid"] for a in audit), "selected_candidates": len(accepted), "feature_records": len(backend.records), "cache_hits": sum(r["cache_hit"] for r in backend.records), "duration_seconds": round(time.monotonic() - started, 3), "score_calibrated": False, "metrics": result}
        write_json(output / "summary.json", summary)
        event("classification", "completed")
        event("delivery", "completed")
        return summary
    except Exception as error:
        event("pipeline", "failed", error_type=type(error).__name__)
        raise

def replay(root, output):
    root, output = Path(root), Path(output)
    rows = read_csv(root / "data/case/generated_20.csv")
    selected, audit = screen(rows)
    paper = list(__import__("csv").DictReader((root / "data/case/candidates_5.csv").open(encoding="utf-8")))
    by_id = {r["molecule_id"]: r for r in selected}
    if len(rows) != 20 or sum(a["valid"] for a in audit) != 7 or len(selected) != 5:
        raise ValueError("Case counts differ from the paper")
    for row in paper:
        current = by_id[row["molecule_id"]]
        if current["smiles"] != row["smiles"] or round(current["qed"], 2) != float(row["qed"]) or round(current["sa"], 2) != float(row["sa"]):
            raise ValueError("Candidate structure or property differs from the paper snapshot")
    write_csv(output / "paper_candidates.csv", paper)
    summary = {"record_type": "historical_case_replay", "historical_replay": True, "new_generation": False, "new_quantum_execution": False, "new_classification": False, "recomputed_steps": ["structure_validation", "QED", "SA"], "counts": {"raw": 20, "valid": 7, "selected": 5}, "score_source": "recorded_paper_scores_not_recomputed"}
    write_json(output / "summary.json", summary)
    write_json(output / "screening.json", audit)
    return summary
