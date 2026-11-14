"""Restore missing public training files, pinned by the bundled SHA-256 manifest."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / "data/generation/upstream.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if ROOT.resolve() not in path.parents:
            raise ValueError("Data path leaves repository")
        if path.exists():
            content = path.read_bytes()
        else:
            if not item["url"].startswith("https://raw.githubusercontent.com/blender-nlp/MolT5/"):
                raise ValueError("Unexpected upstream URL")
            request = urllib.request.Request(item["url"], headers={"User-Agent": "paper-repro-data/1.0"})
            with urllib.request.urlopen(request, timeout=90) as source:
                content = source.read(40 * 1024 * 1024 + 1)
            if len(content) > 40 * 1024 * 1024:
                raise ValueError("Unexpected download size")
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Upstream content changed or local file is corrupt: " + path.name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(path.relative_to(ROOT).as_posix() + " verified")

if __name__ == "__main__":
    main()
