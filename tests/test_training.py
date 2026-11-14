from pathlib import Path
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from s3p_repro.training import read_pairs, tiny_tokenizer, encode_pair, collate_examples, train_generator
from s3p_repro.common import read_json

ROOT = Path(__file__).resolve().parents[1]

def test_description_separator_padding_mask_and_eos_target():
    tokenizer = tiny_tokenizer([{"description": "A molecule", "smiles": "CCO"}])
    example = encode_pair(tokenizer, "A molecule", "CCO", 128)
    first = next(i for i, v in enumerate(example["labels"]) if v != -100)
    assert tokenizer.decode(example["input_ids"][:first]).endswith("SMILES:")
    assert example["labels"][-1] == tokenizer.eos_token_id
    short = encode_pair(tokenizer, "A", "C", 128)
    batch = collate_examples([example, short], tokenizer.pad_token_id)
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()

def test_tiny_training_performs_backward_and_saves_loadable_model(tmp_path):
    config = read_json(ROOT / "configs/paper.json")
    summary = train_generator(ROOT / "data/generation/chebi20/train.txt", tmp_path / "model", config, tiny=True, steps=2, limit=16, device="cpu")
    assert summary["optimizer_steps"] == 2 and summary["smoke_test"] is True
    assert summary["mean_training_loss"] > 0
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "model", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "model", local_files_only=True)
    assert model.config.vocab_size == len(tokenizer)
    from s3p_repro.generation import generate
    config["generation"]["n_generate"] = 2
    rows, generated = generate(tmp_path / "model", tmp_path / "generated", config, prompt="A molecule.\nSMILES:", max_batches=2, max_stalled=2)
    assert generated["generator_training_smoke_test"] is True
    assert generated["generated"] == len(rows) <= 2
    assert generated["batches"] <= 2
    with pytest.raises(ValueError, match="delimiter"):
        generate(tmp_path / "model", tmp_path / "bad", config, prompt="A molecule")
