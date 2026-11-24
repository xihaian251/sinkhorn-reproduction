"""Pre-flight release checks: wheel checksum, raw CSV integrity, no absolute paths.

Usage: python scripts/verify_release.py [--wheel /path/to/residual_sinkhorn-0.2.0.whl]

The wheel is NOT redistributed.  If a local copy is supplied, its SHA-256 is
checked against the author's published value; otherwise the recorded provenance
is printed and the check is skipped.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_WHEEL = "d7099ea4dbad90242bf7318e2235c18cad579e43d3312faa8bfc76e78c854759"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wheel", type=Path, default=None)
    args = p.parse_args()
    if args.wheel:
        actual = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
        if actual != EXPECTED_WHEEL:
            raise SystemExit(f"Wheel SHA-256 mismatch: {actual}")
        print("OK: wheel SHA-256 matches author's published checksum")
    else:
        print("SKIP: no local wheel supplied; expected SHA-256 =", EXPECTED_WHEEL)

    config = json.loads((ROOT / "data" / "processed" / "phase32_config.json").read_text(encoding="utf-8"))
    for name, expected in config["raw_sha256"].items():
        path = ROOT / "data" / "raw" / name
        if not path.exists():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Raw CSV hash mismatch: {name}")
        print(f"OK: {name} sha256 verified")

    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix in (".npz", ".pyc", ".whl"):
            continue
        if path.name == "verify_release.py":
            continue  # this scanner necessarily contains the search tokens themselves
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(tok in line for tok in ("E:\\", "D:\\", "C:\\Users", "E:/ResearchInfrastructure",
                                           "D:/python", "北海")):
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}")
    if offenders:
        print("WARNING: absolute-path/local-identifier strings remain:")
        for o in offenders:
            print("  ", o)
    else:
        print("OK: no machine-specific absolute paths or local identifiers in text files")


if __name__ == "__main__":
    main()
