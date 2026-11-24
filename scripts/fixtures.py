"""Frozen fixtures for the three-way Sinkhorn reproduction.

Loads the exact eight fixtures used in the historical Phase XXIX / XXXI / XXXII
experiments from data/fixtures/fixtures_097post1.npz.  No input is regenerated.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "data" / "fixtures"

FIX_SPEC = {
    "tiny_8x8": (8, 8, 0.5, 1, 2.0, False),
    "wide_32x512": (32, 512, 0.2, 2, 3.0, False),
    "tall_512x32": (512, 32, 0.2, 3, 3.0, False),
    "square_300x300": (300, 300, 0.15, 4, 2.0, False),
    "cold_64x128_0.01": (64, 128, 0.01, 5, 2.0, False),
    "illcond_150x600": (150, 600, 0.25, 6, 6.0, True),
    "moderate_400x100": (400, 100, 0.12, 7, 4.0, False),
    "warmish_90x90": (90, 90, 0.35, 8, 1.0, False),
}
KWARGS = dict(numItermax=3000, stopThr=1e-7)
SEED = 20261129
REPEATS = 17
WARMUP = 3


def load_fixtures():
    z = np.load(FIXTURES / "fixtures_097post1.npz")
    return {
        name: dict(a=z[f"{name}__a"], b=z[f"{name}__b"], M=z[f"{name}__M"],
                   reg=float(z[f"{name}__reg"]), Q=z[f"{name}__Q"])
        for name in FIX_SPEC
    }
