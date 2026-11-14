"""Scan the public release file set without printing sensitive matched values."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "outputs", "models", "checkpoints", ".cache", "build", "dist"}
FORBIDDEN_SUFFIXES = {".pkl", ".pickle", ".joblib", ".safetensors", ".pt", ".pth", ".bin", ".pem", ".key", ".log", ".zip"}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "provider_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "cloud_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "windows_absolute_path": re.compile(r"\b[A-Za-z]:[\\/]"),
    "home_absolute_path": re.compile(r"/(?:Users|home)/[^\s/]+/"),
    "private_network_address": re.compile(r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"),
    "personal_email": re.compile(r"\b[A-Za-z0-9_.+-]+@(?:gmail|hotmail|outlook|qq|foxmail)\.com\b", re.I),
}

def release_files(root=ROOT):
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(p in EXCLUDED_DIRS or p.endswith(".egg-info") for p in relative.parts):
            continue
        yield path

def scan(root=ROOT):
    problems, count = [], 0
    for path in release_files(root):
        count += 1
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith(".env") or ".local." in path.name:
            problems.append({"file": rel, "rule": "private_or_runtime_file"})
            continue
        if path.stat().st_size > 25 * 1024 * 1024:
            problems.append({"file": rel, "rule": "unexpected_large_file"})
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            problems.append({"file": rel, "rule": "unreviewed_binary_file"})
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    problems.append({"file": rel, "line": i, "rule": name})
    return {"files_scanned": count, "problems": problems, "passed": not problems}

if __name__ == "__main__":
    result = scan()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
