"""Text-conditioned GPT-2 SFT; only SMILES and EOS contribute to loss."""
from __future__ import annotations
import csv
import json
import math
import random
from pathlib import Path
import numpy as np
from .common import sha256, write_json

def read_pairs(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    else:
        with path.open(encoding="utf-8-sig", newline="") as f:
            raw = list(csv.DictReader(f, delimiter="\t"))
        rows = [{"description": r["description"], "smiles": r["SMILES"]} for r in raw]
    if not rows or any(not r.get("description") or not r.get("smiles") for r in rows):
        raise ValueError("Training pairs require nonempty description and smiles")
    return rows

def encode_pair(tokenizer, description, smiles, max_length):
    prefix = tokenizer.encode(description.rstrip() + "\nSMILES:", add_special_tokens=False)
    target = tokenizer.encode(" " + smiles, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(target) >= max_length:
        return None
    # Keep the whole SMILES target and the end of the prompt, including its delimiter.
    prefix = prefix[-(max_length - len(target)):]
    ids = prefix + target
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prefix) + target}

def collate_examples(examples, pad_id):
    import torch
    width = max(len(e["input_ids"]) for e in examples)
    return {key: torch.tensor([e[key] + [pad] * (width - len(e[key])) for e in examples], dtype=torch.long) for key, pad in [("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)]}

def tiny_tokenizer(pairs):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    raw = Tokenizer(models.BPE(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=512, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    raw.train_from_iterator((r["description"] + "\nSMILES: " + r["smiles"] for r in pairs), trainer=trainer)
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")

def train_generator(data, output, config, *, tiny=False, steps=None, limit=None, device="auto", local_files_only=False):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, get_cosine_schedule_with_warmup
    output = Path(output)
    if (output / "config.json").exists():
        raise ValueError("Output already contains a model; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    p = config["training"]
    seed = int(p["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    pairs = read_pairs(data)
    original_count = len(pairs)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        pairs = pairs[:limit]
    if tiny:
        tokenizer = tiny_tokenizer(pairs)
        model = AutoModelForCausalLM.from_config(GPT2Config(vocab_size=len(tokenizer), n_layer=2, n_head=2, n_embd=64, n_positions=p["max_length"], bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id))
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["generation"]["base_model"], local_files_only=local_files_only, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config["generation"]["base_model"], local_files_only=local_files_only, trust_remote_code=False)
        model.config.pad_token_id = tokenizer.pad_token_id
    examples = [encode_pair(tokenizer, r["description"], r["smiles"], p["max_length"]) for r in pairs]
    examples = [e for e in examples if e is not None]
    if not examples:
        raise ValueError("No complete SMILES targets fit within max_length")
    chosen = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model.to(chosen)
    model.train()
    epochs = 1 if tiny else int(p["epochs"])
    batch_size = 2 if tiny else int(p["batch_size"])
    accum = 1 if tiny else int(p["gradient_accumulation"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(examples, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=lambda e: collate_examples(e, tokenizer.pad_token_id))
    total_steps = math.ceil(len(loader) / accum) * epochs
    if steps is not None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        total_steps = min(total_steps, steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    warmup = 0 if tiny else min(total_steps, max(50, total_steps // 10))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    optimizer.zero_grad()
    fp16 = chosen.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    done, losses = 0, []
    for _ in range(epochs):
        for i, batch in enumerate(loader):
            group_start = (i // accum) * accum
            divisor = min(accum, len(loader) - group_start)
            with torch.autocast(device_type="cuda" if fp16 else "cpu", enabled=fp16, dtype=torch.float16):
                loss = model(**{k: v.to(chosen) for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss / divisor).backward()
            losses.append(float(loss.detach().cpu()))
            if (i + 1) % accum == 0 or i + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), p["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                done += 1
                if done == 1 or done % 100 == 0 or done == total_steps:
                    progress = {"optimizer_step": done, "total_steps": total_steps, "loss": losses[-1]}
                    with (output / "training_steps.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(progress, allow_nan=False) + "\n")
                if done >= total_steps:
                    break
        if done >= total_steps:
            break
    output.mkdir(parents=True, exist_ok=True)
    model.config._name_or_path = ""
    tokenizer.init_kwargs.pop("name_or_path", None)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    summary = {"schema_version": 1, "smoke_test": tiny, "base_model": "random_tiny_gpt2" if tiny else config["generation"]["base_model"], "data_sha256": sha256(data), "source_pairs": original_count, "selected_pairs": len(pairs), "used_pairs": len(examples), "dropped_long_targets": len(pairs) - len(examples), "epochs_requested": epochs, "optimizer_steps": done, "partial_training": tiny or steps is not None or limit is not None, "training_config": p, "mean_training_loss": float(np.mean(losses)), "device_type": "cuda" if fp16 else "cpu", "loss_mask": "description, SMILES delimiter and padding excluded; SMILES plus EOS included"}
    write_json(output / "training_summary.json", summary)
    return summary
