"""Verify SHA-256 of every frozen fixture array against fixture_hashes.json.

Usage: python scripts/verify_fixtures.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "data" / "fixtures"


def main():
    hashes = json.loads((FIX / "fixture_hashes.json").read_text(encoding="utf-8"))
    z = np.load(FIX / "fixtures_097post1.npz")
    ok = 0
    for name, keys in hashes.items():
        for key, expected in keys.items():
            arr = np.ascontiguousarray(z[f"{name}__{key}"])
            actual = hashlib.sha256(arr.tobytes()).hexdigest()
            if actual != expected:
                raise SystemExit(f"HASH MISMATCH: {name}/{key}\n expected {expected}\n got      {actual}")
            ok += 1
    print(f"OK: {ok}/{ok} frozen fixture arrays match recorded SHA-256 hashes")


if __name__ == "__main__":
    main()
