"""Deterministic built-in fixture for the branch-cut lab.

Layout (12x14 pixel grid, plaquettes are 11x13):

* a + / - vortex pair in the upper half (charges are detected, not
  assumed),
* one unbalanced vortex near the lower border (nearest boundary cost 1),
* a full-height occluded column at ``col = 9`` that splits the valid
  pixels into exactly two connected regions.

Quality is 1.0 everywhere except a smooth decay around each vortex and
0.0 on occluded pixels.
"""

from __future__ import annotations

import numpy as np

from .core import TAU, detect_residues, list_residues, wrap

H = 12
W = 14
MASK_COLUMN = 9
VORTICES = ((2, 3), (2, 7), (9, 4))
FIXTURE_ID = "fixture-v1-12x14-3res-2blocks"


def _vortex_phase(rows, cols, ci, cj, sign=1.0):
    return sign * np.arctan2(rows - (ci + 0.5), cols - (cj + 0.5))


def build_fixture():
    rows, cols = np.mgrid[0:H, 0:W]
    phi = 0.15 * cols
    for k, (ci, cj) in enumerate(VORTICES):
        sign = 1.0 if k % 2 == 0 else -1.0
        phi += _vortex_phase(rows, cols, ci, cj, sign)

    masked = np.zeros((H, W), dtype=bool)
    masked[:, MASK_COLUMN] = True
    wrapped = wrap(phi)

    quality = np.ones((H, W), dtype=np.float64)
    for ci, cj in VORTICES:
        d2 = (rows - ci - 0.5) ** 2 + (cols - cj - 0.5) ** 2
        quality = np.minimum(quality, 0.35 + 0.65 * (d2 / (d2 + 4.0)))
    quality[masked] = 0.0

    charge = detect_residues(wrapped, masked)
    residues = list_residues(charge)
    return {
        "fixture_id": FIXTURE_ID,
        "height": H,
        "width": W,
        "wrapped": wrapped,
        "quality": quality,
        "masked": masked,
        "charge": charge,
        "residues": residues,
    }
