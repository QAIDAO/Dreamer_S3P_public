# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["numpy>=1.24,<2", "spinqit==0.2.4"]
# ///
"""Optional device adapter. Credentials are read only from the environment."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

def execute(request):
    if request.get("backend") != "spinq-nmr" or request.get("qubits") != 3 or request.get("reps") != 1 or request.get("shots") != 128:
        raise ValueError("Only the paper's three-qubit one-layer circuit is supported")
    angles = request.get("angles", [])
    if len(angles) != 3 or not all(math.isfinite(float(x)) for x in angles):
        raise ValueError("Expected three finite angles")
    keys = ["SPINQ_NMR_IP", "SPINQ_NMR_PORT", "SPINQ_NMR_USERNAME", "SPINQ_NMR_PASSWORD"]
    if any(not os.environ.get(k) for k in keys):
        raise ValueError("Configure the four SPINQ_NMR connection variables locally")
    from spinqit import Circuit, Ry, Rz, CX, NMRConfig, get_compiler, get_nmr
    circuit = Circuit()
    q = circuit.allocateQubits(3)
    for i, angle in enumerate(angles):
        circuit << (Ry, q[i], float(angle))
        circuit << (Rz, q[i], float(angle))
    circuit << (CX, (q[0], q[1]))
    circuit << (CX, (q[1], q[2]))
    config = NMRConfig()
    config.configure_shots(128)
    config.configure_ip(os.environ[keys[0]])
    config.configure_port(int(os.environ[keys[1]]))
    config.configure_account(os.environ[keys[2]], os.environ[keys[3]])
    config.configure_task(request["task_name"], "Photoresponse feature acquisition")
    config.configure_measure_qubits([0, 1, 2])
    result = get_nmr().execute(get_compiler("native").compile(circuit, 0), config)
    p = [0.0] * 8
    for bitstring, value in result.probabilities.items():
        p[int(bitstring, 2)] = float(value)
    payload = {k: v for k, v in request.items() if k != "task_name"}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"request_sha256": key, "probabilities": p, "is_hardware": True, "execution_source": "spinq-nmr"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-hardware", action="store_true")
    args = p.parse_args()
    if not args.allow_hardware:
        p.error("Device execution requires explicit --allow-hardware authorization")
    try:
        result = execute(json.loads(args.request.read_text(encoding="utf-8")))
        args.output.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    except Exception as error:
        # SDK exception messages can contain connection details; do not persist them.
        print(json.dumps({"status": "failed_or_unknown", "error_type": type(error).__name__}))
        raise SystemExit(1) from None

if __name__ == "__main__":
    main()
