# Repository instructions

This repository contains an independent research implementation accompanying FDC2026-0145. The commercial desktop application's internal source is outside its scope.

- Read `agent.md` for the scientific execution role. For reproduction, training, screening or quantum feature workflows, use `.agents/skills/dreamer-s3p-paper/SKILL.md`.
- Keep `data/reference/` and the supplied molecular data unchanged. New outputs go under `outputs/` and retain their actual execution source.
- Run `python -m pytest -q`, `python scripts/verify_data.py` and `python scripts/check_release.py` after relevant code or data changes.
- Do not silently replace failed model or device calls with recorded paper results. A real-device submission requires explicit user authorization; an unknown submission is reconciled before retrying.
- Do not store credentials, private connection details or local absolute paths in committed files.
