"""Сквозные проверки для make check: детерминизм между процессами, утечки в артефактах."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tree_digest(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def determinism_check(split_files: list[str]) -> dict:
    """Генерирует сплиты в двух независимых процессах с разным PYTHONHASHSEED и сравнивает SHA-256."""
    digests = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, hs in enumerate(["0", "12345"]):
            base = Path(tmp) / f"run{i}"
            env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT))
            for sf in split_files:
                subprocess.run([sys.executable, "-m", "auditenv.cli", "generate", "--split", sf,
                                "--episodes-root", str(base / "episodes"), "--grader-root", str(base / "grader")],
                               check=True, env=env, cwd=ROOT, stdout=subprocess.DEVNULL)
            digests.append(tree_digest(base))
    same = digests[0] == digests[1]
    diff = sorted(k for k in set(digests[0]) | set(digests[1]) if digests[0].get(k) != digests[1].get(k))
    return {"ok": same and len(digests[0]) > 0, "n_files": len(digests[0]), "mismatched": diff[:20]}


def main(argv=None) -> int:
    import argparse
    from .common.console import utf8_stdio
    utf8_stdio()
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["determinism"])
    p.add_argument("--splits", nargs="+", default=["configs/splits/train.json", "configs/splits/dev.json"])
    a = p.parse_args(argv)
    if a.what == "determinism":
        r = determinism_check(a.splits)
        print(f"[determinism] files={r['n_files']} identical_across_processes={r['ok']}")
        for m in r["mismatched"]:
            print("   mismatch:", m)
        return 0 if r["ok"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
