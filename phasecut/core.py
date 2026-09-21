"""Core phase numerics.

Data conventions (数据口径)
---------------------------
* Wrapped phase differences are reduced to the half-open interval
  ``(-pi, pi]``: a difference of exactly ``+pi`` stays ``+pi``, a difference
  of exactly ``-pi`` is mapped to ``+pi``.  The reduction uses only
  multiply/floor/subtract (no ``atan2``/``fmod``), so the endpoint decision
  cannot flip with platform libm rounding.
* Residue charge of a plaquette is ``round(sum(wrapped diffs) / 2*pi)``
  walking the 4-pixel loop right -> down -> left -> up (clockwise in image
  coordinates).  Positive sum => positive charge.
* Occluded pixels (``valid == False``) never take part in a 4-pixel loop and
  must never be crossed by a branch cut.
* Unwrapping is per connected component: each component only determines its
  own integer ``2*pi`` offset; no global minimum is assumed across components.
"""
from __future__ import annotations

import math

import numpy as np

PI = math.pi
TWO_PI = 2.0 * math.pi

# BFS neighbour order: up, right, down, left (deterministic replay).
NEIGHBOURS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def wrap_phase(delta):
    """Reduce phase difference(s) to the half-open interval ``(-pi, pi]``.

    Exact ``-pi`` maps to ``+pi``; exact ``+pi`` stays ``+pi``.  Implemented
    with ``floor`` only so the endpoint rule is identical on every machine.
    """
    d = np.asarray(delta, dtype=np.float64)
    w = d - TWO_PI * np.floor((d + PI) / TWO_PI)
    # w is in [-pi, pi); move the closed -pi endpoint to +pi -> (-pi, pi]
    w = np.where(w <= -PI, w + TWO_PI, w)
    if w.ndim == 0:
        return float(w)
    return w


def detect_residues(wrapped, valid):
    """Return (charges, computed) plaquette arrays of shape (H-1, W-1).

    ``charges[r, c]`` in {-1, 0, +1} is the residue charge of the plaquette
    whose top-left pixel is (r, c).  ``computed`` is False for plaquettes
    touching an occluded pixel; those plaquettes carry no charge.
    """
    wrapped = np.asarray(wrapped, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    h, w = wrapped.shape
    charges = np.zeros((h - 1, w - 1), dtype=np.int64)
    computed = np.zeros((h - 1, w - 1), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            if not (valid[r, c] and valid[r, c + 1]
                    and valid[r + 1, c] and valid[r + 1, c + 1]):
                continue
            d1 = wrap_phase(wrapped[r, c + 1] - wrapped[r, c])
            d2 = wrap_phase(wrapped[r + 1, c + 1] - wrapped[r, c + 1])
            d3 = wrap_phase(wrapped[r + 1, c] - wrapped[r + 1, c + 1])
            d4 = wrap_phase(wrapped[r, c] - wrapped[r + 1, c])
            charges[r, c] = int(round((d1 + d2 + d3 + d4) / TWO_PI))
            computed[r, c] = True
    return charges, computed


def residue_list(wrapped, valid):
    """Residues as a list of dicts with stable ids ``r0, r1, ...``."""
    charges, _ = detect_residues(wrapped, valid)
    out = []
    for (r, c), q in np.ndenumerate(charges):
        if q != 0:
            out.append({"id": f"r{len(out)}", "r": int(r), "c": int(c),
                        "charge": int(q)})
    return out


def norm_edge(a, b):
    """Normalise a pixel edge ((r1,c1),(r2,c2)) to a canonical ordering."""
    return (a, b) if a <= b else (b, a)


def unwrap(wrapped, valid, blocked_edges, start):
    """Flood unwrap from ``start`` without crossing ``blocked_edges``.

    ``blocked_edges`` is an iterable of normalised pixel edges.  Every
    connected component is unwrapped from its own seed anchored at its own
    wrapped value, so each component only fixes its own integer ``2*pi``
    offset.  Returns (unwrapped, labels, components).
    """
    wrapped = np.asarray(wrapped, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    h, w = wrapped.shape
    sr, sc = int(start[0]), int(start[1])
    if not (0 <= sr < h and 0 <= sc < w):
        raise ValueError("start pixel outside grid")
    if not valid[sr, sc]:
        raise ValueError("start pixel is occluded")

    blocked = set()
    for a, b in blocked_edges:
        blocked.add(norm_edge(tuple(a), tuple(b)))

    labels = np.full((h, w), -1, dtype=np.int64)
    unwrapped = np.full((h, w), np.nan)
    components = []

    def flood(seed):
        cid = len(components)
        r0, c0 = seed
        labels[r0, c0] = cid
        unwrapped[r0, c0] = wrapped[r0, c0]
        queue = [(r0, c0)]
        size = 0
        while queue:
            r, c = queue.pop()
            size += 1
            for dr, dc in NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if not valid[nr, nc] or labels[nr, nc] >= 0:
                    continue
                if norm_edge((r, c), (nr, nc)) in blocked:
                    continue
                labels[nr, nc] = cid
                unwrapped[nr, nc] = (unwrapped[r, c]
                                     + wrap_phase(wrapped[nr, nc]
                                                  - wrapped[r, c]))
                queue.append((nr, nc))
        components.append({"id": cid, "seed": [r0, c0], "size": size})

    flood((sr, sc))
    start_label = 0
    for r in range(h):
        for c in range(w):
            if valid[r, c] and labels[r, c] < 0:
                flood((r, c))
    for comp in components:
        comp["reachable_from_start"] = comp["id"] == start_label
    return unwrapped, labels, components
