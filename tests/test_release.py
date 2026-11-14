from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_check", ROOT / "scripts/check_release.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_scanner_redacts_matches_and_excludes_runtime_outputs(tmp_path):
    (tmp_path / "outputs").mkdir()
    sensitive = "sk" + "-" + "proj" + "-" + "A" * 40
    (tmp_path / "outputs/runtime.txt").write_text(sensitive)
    assert module.scan(tmp_path)["passed"]
    (tmp_path / "public.txt").write_text(sensitive)
    result = module.scan(tmp_path)
    assert result["passed"] is False
    assert result["problems"][0]["rule"] == "provider_key"
    assert sensitive not in str(result)
