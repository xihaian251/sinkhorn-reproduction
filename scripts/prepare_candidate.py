"""Build B (hardened Candidate A) from a pristine POT 0.9.7.post1 source tree.

This script fixes the historical build order: the patch is applied to the
*stock* file, the patched result becomes ``_sinkhorn_candidate.py`` (B), and the
stock file (A) is restored byte-for-byte afterwards.  It therefore guarantees:

    A = original ot/bregman/_sinkhorn.py
    B = patched ot/bregman/_sinkhorn_candidate.py

It validates the POT version and stock source identity first, runs
``git apply --check``, applies the patch, copies the patched source to the
candidate module, restores stock, and verifies both source hashes against the
values recorded in data/processed/phase32_config.json.  If anything fails, the
stock file is restored in a ``finally`` block so a failed run never leaves a
wrong source state.

Line endings: the historical trees used CRLF while a fresh POSIX-style checkout
usually uses LF.  Comparisons are performed after normalizing to CRLF, which is
an explicitly recorded, semantics-preserving normalization.

Usage (from repository root)::

    python scripts/prepare_candidate.py --pot-src /path/to/POT-0.9.7.post1
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH = ROOT / "patches" / "POT_PHASE29_CANDIDATE_A_HARDENED.patch"
CONFIG = ROOT / "data" / "processed" / "phase32_config.json"

HIST_A = "71ee6030e5dffc4dff9b0200a8d2f2317da5e2546102e0f478d1b247bafe2592"
HIST_B = "586f71bb1a80af5260139df663387df96f301d7a5034f6a2ea2975c75e87ea61"
EXPECTED_VERSION = "0.9.7.post1"


def crlf_hash(data: bytes) -> str:
    """SHA-256 after normalizing newlines to CRLF (historical checkout convention)."""
    normalized = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(normalized).hexdigest()


def read_version(pot_src: Path) -> str:
    init = pot_src / "ot" / "__init__.py"
    if not init.is_file():
        raise RuntimeError(f"not a POT source tree: {pot_src} (missing ot/__init__.py)")
    text = init.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not m:
        raise RuntimeError("cannot determine POT version from ot/__init__.py")
    return m.group(1)


def check_config_hashes():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = cfg.get("source_sha256", {})
    if expected.get("A") != HIST_A:
        raise RuntimeError("phase32_config.json records a different A hash than this script")
    return cfg


def run_git(args, cwd):
    result = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pot-src", required=True, type=Path,
                   help="POT 0.9.7.post1 source tree (with .git, containing ot/)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing candidate module")
    args = p.parse_args()
    pot_src = args.pot_src.resolve()

    check_config_hashes()
    version = read_version(pot_src)
    print(f"POT version: {version}")
    if version != EXPECTED_VERSION:
        raise SystemExit(f"Expected {EXPECTED_VERSION}, got {version}")

    if not (pot_src / ".git").exists():
        raise SystemExit("--pot-src must be a git work tree (contains .git) for git apply")
    if not PATCH.is_file():
        raise SystemExit(f"patch not found: {PATCH}")

    bregman = pot_src / "ot" / "bregman"
    stock = bregman / "_sinkhorn.py"
    candidate = bregman / "_sinkhorn_candidate.py"

    if not stock.is_file():
        raise SystemExit(f"stock source missing: {stock}")
    stock_original = stock.read_bytes()
    if crlf_hash(stock_original) != HIST_A:
        raise SystemExit(
            "stock source identity mismatch: normalized SHA-256 is "
            f"{crlf_hash(stock_original)}, expected {HIST_A}\n"
            "Refusing to patch an unrecognized POT source tree.")

    if candidate.exists() and not args.force:
        if crlf_hash(candidate.read_bytes()) == HIST_B:
            print("candidate already present and correct; nothing to do")
            return
        raise SystemExit(
            f"candidate exists but does not match the historical B hash; "
            f"re-run with --force to overwrite: {candidate}")

    # Keep the pristine stock bytes in memory and on disk until the very end.
    with tempfile.TemporaryDirectory() as td:
        backup = Path(td) / "_sinkhorn.py.orig"
        backup.write_bytes(stock_original)
        stock_restored = False
        try:
            run_git(["apply", "--check", str(PATCH)], pot_src)
            print("git apply --check: OK")
            run_git(["apply", str(PATCH)], pot_src)
            print("git apply: OK")

            patched = stock.read_bytes()
            if crlf_hash(patched) != HIST_B:
                raise RuntimeError(
                    f"patched source hash {crlf_hash(patched)} != expected {HIST_B}")

            # Write candidate with normalized CRLF so its on-disk hash equals HIST_B.
            candidate.write_bytes(patched.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
            print(f"wrote {candidate.relative_to(pot_src)}")

            shutil.copyfile(backup, stock)
            stock_restored = True
            print("stock file restored")
        finally:
            if not stock_restored:
                shutil.copyfile(backup, stock)
                print("stock file restored after failure", file=sys.stderr)

    a_hash = crlf_hash(stock.read_bytes())
    b_hash = crlf_hash(candidate.read_bytes())
    if a_hash != HIST_A or b_hash != HIST_B:
        raise SystemExit(
            f"final identity check FAILED: A={a_hash} B={b_hash}")
    print(f"OK: A normalized sha256 = {a_hash}")
    print(f"OK: B normalized sha256 = {b_hash}")
    print("A/B identity verified; build complete")


if __name__ == "__main__":
    main()
