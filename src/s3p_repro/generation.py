from __future__ import annotations
from pathlib import Path
from .common import write_csv, write_json, read_json

def generate(model_path, output, config, *, prompt=None, max_batches=20, max_stalled=5):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path, output = Path(model_path), Path(output)
    if not (model_path / "config.json").is_file():
        raise ValueError("Provide a locally trained conditional generator checkpoint")
    p = config["generation"]
    prompt = prompt if prompt is not None else p["prompt"]
    if not prompt.rstrip().endswith("SMILES:"):
        raise ValueError("Conditional generation requires a trailing SMILES: delimiter")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    torch.manual_seed(p["seed"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokens = tokenizer(prompt, return_tensors="pt")
    prefix_length = tokens["input_ids"].shape[1]
    if prefix_length >= p["max_length"]:
        raise ValueError("Prompt already reaches the configured total max_length")
    strings, seen = [], set()
    batches = stalled = 0
    while len(strings) < p["n_generate"] and batches < max_batches and stalled < max_stalled:
        n = min(20, p["n_generate"] - len(strings))
        batch = tokenizer([prompt] * n, return_tensors="pt", padding=True).to(device)
        before = len(strings)
        with torch.no_grad():
            samples = model.generate(**batch, max_length=p["max_length"], do_sample=True, temperature=p["temperature"], top_p=p["top_p"], pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for sequence in samples:
            raw = tokenizer.decode(sequence[prefix_length:], skip_special_tokens=True).strip()
            if raw and raw not in seen:
                seen.add(raw)
                strings.append(raw)
        batches += 1
        stalled = stalled + 1 if len(strings) == before else 0
    rows = [{"molecule_id": f"G{i:02d}", "smiles": s} for i, s in enumerate(strings, 1)]
    reason = "target_reached" if len(rows) == p["n_generate"] else ("stalled" if stalled >= max_stalled else "max_batches")
    write_csv(output / "generated.csv", rows)
    training_file = model_path / "training_summary.json"
    smoke = read_json(training_file).get("smoke_test") if training_file.exists() else None
    summary = {"execution_source": "local_conditional_generator", "historical_replay": False, "generator_training_smoke_test": smoke, "requested": p["n_generate"], "generated": len(rows), "batches": batches, "stop_reason": reason, "generation_parameters": p, "prompt": prompt, "max_length_includes_prompt": True}
    write_json(output / "generation_summary.json", summary)
    return rows, summary
