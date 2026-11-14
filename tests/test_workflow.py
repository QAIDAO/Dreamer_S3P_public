from pathlib import Path
import sys
import pytest
from s3p_repro.common import read_json
from s3p_repro.pipeline import replay
from s3p_repro.quantum import FeatureBackend, SubmissionUnknown

ROOT = Path(__file__).resolve().parents[1]

def test_replay_is_explicit_and_checks_paper_properties(tmp_path):
    result = replay(ROOT, tmp_path)
    assert result["counts"] == {"raw": 20, "valid": 7, "selected": 5}
    assert result["new_quantum_execution"] is False
    assert result["new_classification"] is False
    assert not (tmp_path / "feature_cache").exists()

def test_completed_features_are_reused_without_new_sampling(tmp_path):
    backend = FeatureBackend(tmp_path, sample_shots=True)
    a = backend.execute([0.1, 0.2, 0.3], "first")
    b = backend.execute([0.1, 0.2, 0.3], "second")
    assert (a == b).all()
    assert backend.records[-1]["cache_hit"] is True
    assert backend.records[-1]["molecule_id"] == "second"
    assert backend.records[-1]["is_hardware"] is False

def test_unknown_device_submission_never_automatically_retries(tmp_path):
    stub = tmp_path / "failing_worker.py"
    counter = tmp_path / "invocations.txt"
    stub.write_text("from pathlib import Path\np=Path(__file__).with_name('invocations.txt')\np.write_text((p.read_text() if p.exists() else '')+'called\\n')\nraise SystemExit(1)\n", encoding="utf-8")
    backend = FeatureBackend(tmp_path / "cache", name="spinq-nmr", worker_python=sys.executable, worker_script=stub, allow_hardware=True)
    with pytest.raises(SubmissionUnknown):
        backend.execute([0.1, 0.2, 0.3], "m1")
    with pytest.raises(SubmissionUnknown):
        backend.execute([0.1, 0.2, 0.3], "m1")
    assert counter.read_text().splitlines() == ["called"]
    records = [read_json(p) for p in (tmp_path / "cache").glob("*.json") if not p.name.endswith(".request.json")]
    assert any(r.get("status") == "unknown" for r in records)

def test_hardware_requires_explicit_authorization(tmp_path):
    with pytest.raises(ValueError):
        FeatureBackend(tmp_path, name="spinq-nmr")
