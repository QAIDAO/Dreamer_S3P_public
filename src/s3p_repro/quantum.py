from __future__ import annotations
import math
import subprocess
from pathlib import Path
import numpy as np
from .common import digest, read_json, write_json

def validate_probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.shape != (8,) or not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("Expected eight finite, nonnegative probabilities with positive sum")
    return p / p.sum()

def simulate(angles, reps=1):
    angles = np.asarray(angles, dtype=float)
    if angles.shape != (3,) or not np.all(np.isfinite(angles)):
        raise ValueError("Expected three finite angles")
    if reps != 1:
        raise ValueError("This paper implementation uses exactly one layer")
    state = np.zeros(8, dtype=complex)
    state[0] = 1
    for q, theta in enumerate(angles):
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        gate = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]) @ np.array([[c, -s], [s, c]])
        for i in range(8):
            if not i & (1 << q):
                j = i | (1 << q)
                state[i], state[j] = gate @ state[[i, j]]
    for control, target in [(0, 1), (1, 2)]:
        for i in range(8):
            if i & (1 << control) and not i & (1 << target):
                j = i | (1 << target)
                state[i], state[j] = state[j], state[i]
    return validate_probabilities(np.abs(state) ** 2)

class SubmissionUnknown(RuntimeError):
    pass

class FeatureBackend:
    def __init__(self, cache, *, name="simulator", shots=128, seed=42, sample_shots=False, worker_python=None, worker_script=None, allow_hardware=False, retry_unknown=False, timeout=3600):
        if name not in ("simulator", "spinq-nmr"):
            raise ValueError("Unsupported backend")
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be positive")
        if name == "spinq-nmr" and (not allow_hardware or not worker_python or not worker_script):
            raise ValueError("Hardware requires explicit authorization and an isolated worker")
        self.cache, self.name = Path(cache), name
        self.shots, self.seed, self.sample_shots = shots, seed, sample_shots
        self.worker_python, self.worker_script = worker_python, worker_script
        self.retry_unknown, self.timeout = retry_unknown, timeout
        self.records = []

    def execute(self, angles, molecule_id):
        values = np.asarray(angles, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Expected three finite angles")
        request = {"schema_version": 1, "backend": self.name, "qubits": 3, "reps": 1, "shots": self.shots, "angles": values.tolist(), "seed": self.seed, "sample_shots": self.sample_shots}
        key = digest(request)
        self.cache.mkdir(parents=True, exist_ok=True)
        record_path = self.cache / f"{key}.json"
        record = read_json(record_path) if record_path.exists() else {}
        if record.get("status") == "completed":
            p = validate_probabilities(record["probabilities"])
            meta = {"molecule_id": molecule_id, "request_sha256": key, "execution_source": self.name, "is_hardware": self.name == "spinq-nmr", "cache_hit": True}
            self.records.append(meta)
            return p
        if self.name == "spinq-nmr" and record.get("status") in ("submitting", "unknown") and not self.retry_unknown:
            raise SubmissionUnknown("Reconcile the device queue before explicitly retrying this request")
        task_name = f"s3p-{key[:24]}"
        if self.name == "simulator":
            p = simulate(values)
            if self.sample_shots:
                rng = np.random.default_rng(int(digest([self.seed, key])[:16], 16))
                p = rng.multinomial(self.shots, p) / self.shots
            mode = "multinomial_sampling" if self.sample_shots else "analytic_statevector"
        else:
            write_json(record_path, {"status": "submitting", "task_name": task_name, "request_sha256": key})
            request_file = self.cache / f"{key}.request.json"
            response_file = self.cache / f"{key}.response.json"
            write_json(request_file, {**request, "task_name": task_name})
            try:
                subprocess.run([self.worker_python, str(self.worker_script), "--request", str(request_file), "--output", str(response_file), "--allow-hardware"], check=True, timeout=self.timeout, capture_output=True, text=True)
                reply = read_json(response_file)
                if reply.get("request_sha256") != key or reply.get("is_hardware") is not True:
                    raise ValueError("Hardware reply does not match its request")
                p = validate_probabilities(reply["probabilities"])
                mode = "hardware_readout"
            except Exception as error:
                write_json(record_path, {"status": "unknown", "task_name": task_name, "request_sha256": key, "error_type": type(error).__name__})
                raise SubmissionUnknown("Submission status unknown; inspect the queue before retrying") from None
        write_json(record_path, {"status": "completed", "task_name": task_name, "request_sha256": key, "execution_source": self.name, "is_hardware": self.name == "spinq-nmr", "mode": mode, "probabilities": p.tolist()})
        self.records.append({"molecule_id": molecule_id, "request_sha256": key, "execution_source": self.name, "is_hardware": self.name == "spinq-nmr", "cache_hit": False})
        return p

    def embed(self, angles, rows):
        if len(angles) != len(rows):
            raise ValueError("Molecule/feature row count mismatch")
        return np.asarray([self.execute(a, r["molecule_id"]) for a, r in zip(angles, rows)])
