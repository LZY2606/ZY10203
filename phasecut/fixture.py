"""Deterministic fixture dataset.

16x20 grid.  Column 10 is fully occluded, splitting the valid area into a
left block (cols 0-9) and a right block (cols 11-19).  The wrapped phase is
the reduction of a smooth field plus three point vortices:

* +1 residue at plaquette (4, 3)   (left block, paired)
* -1 residue at plaquette (10, 5)  (left block, paired)
* +1 residue at plaquette (7, 15)  (right block, unbalanced -> boundary)

Quality is low near the vortices and zero inside the occluded band.
Everything is a closed-form function of the pixel coordinates, so the
fixture is bit-identical on every machine and every run.
"""
from __future__ import annotations

import numpy as np

from .core import wrap_phase

SHAPE = (16, 20)
OCCLUDED_COL = 10
VORTICES = ((1.0, 4.5, 3.5), (-1.0, 10.5, 5.5), (1.0, 7.5, 15.5))


def build_fixture():
    h, w = SHAPE
    valid = np.ones((h, w), dtype=bool)
    valid[:, OCCLUDED_COL] = False

    rows, cols = np.mgrid[0:h, 0:w].astype(np.float64)
    field = 0.15 * rows + 0.25 * cols
    for charge, vr, vc in VORTICES:
        field = field + charge * np.arctan2(rows - vr, cols - vc)
    wrapped = wrap_phase(field)

    quality = 0.85 + 0.05 * np.sin(rows * 1.7) * np.cos(cols * 2.3)
    for _, vr, vc in VORTICES:
        dist2 = (rows - vr) ** 2 + (cols - vc) ** 2
        quality = np.where(dist2 < 4.0, 0.15, quality)
    quality = np.where(valid, quality, 0.0)

    return {
        "shape": [h, w],
        "wrapped": wrapped,
        "quality": quality,
        "valid": valid,
    }
