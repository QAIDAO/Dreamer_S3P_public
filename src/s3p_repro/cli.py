from __future__ import annotations
import argparse
import json
from pathlib import Path
from .common import read_json, read_csv, write_json, write_csv

def main():
    parser = argparse.ArgumentParser(description="FDC2026-0145 research workflow")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-data")
    inspect.add_argument("--output", type=Path)
    replay = commands.add_parser("replay-case")
    replay.add_argument("--output", type=Path, default=Path("outputs/replay"))
    plan = commands.add_parser("plan")
    plan.add_argument("--candidates", type=Path)
    run = commands.add_parser("pipeline")
    run.add_argument("--output", type=Path, default=Path("outputs/simulator"))
    run.add_argument("--candidates", type=Path)
    run.add_argument("--generator-model", type=Path)
    run.add_argument("--backend", choices=["simulator", "spinq-nmr"], default="simulator")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--sample-shots", action="store_true")
    run.add_argument("--worker-python")
    run.add_argument("--allow-hardware", action="store_true")
    run.add_argument("--retry-unknown", action="store_true")
    train = commands.add_parser("train-generator")
    train.add_argument("--data", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--tiny", action="store_true")
    train.add_argument("--steps", type=int)
    train.add_argument("--limit", type=int)
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train.add_argument("--local-files-only", action="store_true")
    gen = commands.add_parser("generate")
    gen.add_argument("--model", type=Path, required=True)
    gen.add_argument("--output", type=Path, default=Path("outputs/generation"))
    gen.add_argument("--prompt")
    filt = commands.add_parser("filter")
    filt.add_argument("--input", type=Path, required=True)
    filt.add_argument("--output", type=Path, default=Path("outputs/filter"))
    args = parser.parse_args()
    config = read_json(args.root / "configs/paper.json")
    expected = [config["fingerprint"]["radius"], config["fingerprint"]["bits"], config["preprocessing"]["pca_components"], config["preprocessing"]["angle_min"], config["preprocessing"]["angle_max"], config["circuit"]["reps"], config["circuit"]["shots"], config["classifier"]["C"]]
    if expected != [2, 128, 3, 0.0, 3.141592653589793, 1, 128, 1.0]:
        parser.error("This reference implementation fixes the paper's fingerprint, PCA, angle, circuit and SVM parameters")
    if args.command == "inspect-data":
        from .chemistry import describe_split
        result = describe_split(read_csv(args.root / "data/classification/train_648.csv"), read_csv(args.root / "data/classification/test_162.csv"))
        if args.output:
            write_json(args.output, result)
    elif args.command == "plan":
        from .chemistry import screen
        raw = read_csv(args.candidates or args.root / "data/case/generated_20.csv")
        selected, audit = screen(raw)
        result = {"train_samples": 648, "test_samples": 162, "candidate_samples": len(selected), "maximum_uncached_circuits": 648 + 162 + len(selected), "shots_per_circuit": 128, "reps": 1, "hardware_submitted": False, "generation": "provided_smiles_snapshot", "valid_structures": sum(a["valid"] for a in audit)}
    elif args.command == "replay-case":
        from .pipeline import replay
        result = replay(args.root, args.output)
    elif args.command == "pipeline":
        from .pipeline import run
        result = run(args.root, args.output, config, backend_name=args.backend, seed=args.seed, sample_shots=args.sample_shots, candidate_file=args.candidates, generator_model=args.generator_model, worker_python=args.worker_python, allow_hardware=args.allow_hardware, retry_unknown=args.retry_unknown)
    elif args.command == "train-generator":
        from .training import train_generator
        data = args.data or args.root / "data/generation/chebi20/train.txt"
        result = train_generator(data, args.output, config, tiny=args.tiny, steps=args.steps, limit=args.limit, device=args.device, local_files_only=args.local_files_only)
    elif args.command == "generate":
        from .generation import generate
        _, result = generate(args.model, args.output, config, prompt=args.prompt)
    else:
        from .chemistry import screen
        selected, audit = screen(read_csv(args.input), config["screen"]["qed_min"], config["screen"]["sa_max"])
        write_csv(args.output / "filtered.csv", selected)
        write_json(args.output / "screening.json", audit)
        result = {"raw_strings": len(audit), "valid_structures": sum(a["valid"] for a in audit), "selected_candidates": len(selected)}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
