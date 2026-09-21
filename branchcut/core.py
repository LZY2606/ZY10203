"""Phase wrapping, residue (charge) detection, branch cuts and unwrapping.

Coordinates
----------
Arrays use image coordinates ``(row, col)``.

* A *plaquette* is a 2x2 cell loop whose top-left pixel is ``(i, j)``;
  plaquettes live on an ``(H-1) x (W-1)`` grid.
* A residue (charge) sits at a plaquette and equals +1 or -1 when the
  wrapped sum of the four edge phase differences is a full +/-2pi turn.
* A branch cut links residues to each other or to a *boundary cell*
  (virtual plaquette ``br-0`` .. ``br-(2H+2W-5)``) and blocks pixel edges
  during flood-fill unwrapping.

Wrap convention
---------------
Wrapped phases live in the half-open interval ``[-pi, +pi)``.  The
reduction uses ``floor((x + pi) / (2pi))`` so that input equal to either
endpoint maps deterministically to ``-pi`` regardless of floating point
rounding; there is no value equal to ``+pi`` in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

TAU = 2.0 * math.pi
HALF_PI = math.pi


def wrap(phi):
    """Wrap phases into the half-open interval ``[-pi, pi)``.

    Works for scalars and NumPy arrays.  Both ``+pi`` and ``-pi`` map
    exactly to ``-pi``; the convention is independent of platform
    rounding.
    """
    arr = np.asarray(phi, dtype=np.float64)
    reduced = arr - TAU * np.floor((arr + math.pi) / TAU)
    if np.isscalar(phi) and arr.ndim == 0:
        return float(reduced)
    return reduced


def _wrap_diff(a, b):
    """Wrapped difference ``a - b`` in ``[-pi, pi)`` (half-open)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return d - TAU * np.floor((d + math.pi) / TAU)


def detect_residues(wrapped: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Return an ``(H-1, W-1)`` charge array (-1/0/+1).

    A plaquette is skipped (charge 0) if any of its four pixels is
    masked: occluded cells never take part in a four-edge loop.

    The loop is traversed bottom-left -> bottom-right -> top-right ->
    top-left -> bottom-left (clockwise for ``row`` pointing down),
    summing wrapped edge differences.  A total of ``+/- 2pi`` (within
    the robust ``pi`` half-period threshold) yields charge +/-1.
    """
    w = np.asarray(wrapped, dtype=np.float64)
    m = np.asarray(masked, dtype=bool)
    h, width = w.shape
    charge = np.zeros((h - 1, width - 1), dtype=np.int8)

    top_left = w[:-1, :-1]
    top_right = w[:-1, 1:]
    bot_left = w[1:, :-1]
    bot_right = w[1:, 1:]

    loop = (
        _wrap_diff(bot_left, bot_right)
        + _wrap_diff(bot_right, top_right)
        + _wrap_diff(top_right, top_left)
        + _wrap_diff(top_left, bot_left)
    )
    valid = ~(m[:-1, :-1] | m[:-1, 1:] | m[1:, :-1] | m[1:, 1:])
    total = np.where(valid, loop, 0.0)
    charge[total > math.pi] = 1
    charge[total < -math.pi] = -1
    return charge


@dataclass(frozen=True)
class Residue:
    rid: str
    i: int
    j: int
    charge: int


def list_residues(charge: np.ndarray) -> list[Residue]:
    out: list[Residue] = []
    for idx, c in enumerate(charge.ravel()):
        if c:
            i, j = np.unravel_index(idx, charge.shape)
            out.append(Residue(f"r{len(out)}", int(i), int(j), int(c)))
    return out


def boundary_cell_ids(ph: int, pw: int) -> list[str]:
    """Virtual boundary-cell ids for a ``ph x pw`` plaquette grid."""
    ids: list[str] = []
    for j in range(pw):
        ids.append(f"br-{j}")                    # top (above row 0)
    for i in range(ph - 1):
        ids.append(f"br-{pw + i}")               # right
    for j in range(pw):
        ids.append(f"br-{pw + ph - 1 + (pw - 1 - j)}")  # bottom
    for i in range(ph - 1):
        ids.append(f"br-{2 * pw + ph - 1 + (ph - 2 - i)}")  # left
    return ids


def parse_node(node_id: str):
    if node_id.startswith("br-"):
        return ("boundary", int(node_id[3:]))
    body = node_id.split(":", 1)[1] if ":" in node_id else node_id[1:]
    i, j = body.split("_")
    return ("cell", int(i), int(j))


def boundary_distance(i: int, j: int, ph: int, pw: int) -> int:
    """Manhattan steps from a plaquette to the nearest virtual cell."""
    return min(i, j, ph - 1 - i, pw - 1 - j)


def _bfs_cell_path(src, dst, blocked_cells, ph, pw):
    """Shortest 4-neighbour path over valid plaquettes; deterministic."""
    from collections import deque

    prev = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        ci, cj = cur
        for di, dj in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < ph and 0 <= nj < pw):
                continue
            nxt = (ni, nj)
            if nxt in prev or nxt in blocked_cells:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if dst not in prev:
        return None
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def boundary_positions(ph: int, pw: int) -> dict[str, tuple]:
    """Map boundary cell id to ``(side, along_index)``."""
    pos: dict[str, tuple] = {}
    for j in range(pw):
        pos[f"br-{j}"] = ("top", j)
    for i in range(ph - 1):
        pos[f"br-{pw + i}"] = ("right", i)
    for j in range(pw):
        pos[f"br-{2 * pw + ph - 2 - j}"] = ("bottom", j)
    for i in range(ph - 1):
        pos[f"br-{2 * pw + 2 * ph - 3 - i}"] = ("left", i)
    return pos


def _edge_key(r1, c1, r2, c2):
    a = (r1, c1) if (r1, c1) <= (r2, c2) else (r2, c2)
    b = (r2, c2) if a == (r1, c1) else (r1, c1)
    return (a[0], a[1], b[0], b[1])


def _blocked_edges_for_move(a, b):
    """Pixel edge crossed when stepping between adjacent plaquettes."""
    (i1, j1), (i2, j2) = a, b
    if j2 == j1 + 1:  # side-by-side: shared vertical pixel edge
        return {_edge_key(i1, j1 + 1, i1 + 1, j1 + 1)}
    if j2 == j1 - 1:
        return {_edge_key(i1, j1, i1 + 1, j1)}
    if i2 == i1 + 1:  # stacked: shared horizontal pixel edge
        return {_edge_key(i1 + 1, j1, i1 + 1, j1 + 1)}
    return {_edge_key(i1, j1, i1, j1 + 1)}


def blocked_plaquettes(masked: np.ndarray) -> set[tuple[int, int]]:
    m = np.asarray(masked, dtype=bool)
    h, width = m.shape
    bad = m[:-1, :-1] | m[:-1, 1:] | m[1:, :-1] | m[1:, 1:]
    return {tuple(x) for x in np.argwhere(bad)}


def _boundary_candidates(i, j, ph, pw):
    """``(virtual_id, side, adj_plaquette, outer_pixel_edge)`` list."""
    cands = []
    if i == 0:
        cands.append((f"br-{j}", "top", (0, j), _edge_key(0, j, 0, j + 1)))
    if i == ph - 1:
        cands.append((f"br-{2*pw+ph-2-j}", "bottom", (ph - 1, j),
                      _edge_key(ph, j, ph, j + 1)))
    if j == 0:
        cands.append((f"br-{2*pw+2*ph-3-i}", "left", (i, 0),
                      _edge_key(i, 0, i + 1, 0)))
    if j == pw - 1:
        cands.append((f"br-{pw+i}", "right", (i, pw - 1),
                      _edge_key(i, pw, i + 1, pw)))
    return cands


def _nearest_boundary(src, blocked, ph, pw):
    from collections import deque

    dist = {src: 0}
    q = deque([src])
    best = None  # (cost, order, side_plq, virtual_id, outer_edge)
    side_order = {"top": 0, "right": 1, "bottom": 2, "left": 3}
    while q:
        cur = q.popleft()
        ci, cj = cur
        for vid, side, plq, outer in _boundary_candidates(ci, cj, ph, pw):
            order = side_order[side] * (ph + pw) + ci + cj
            cand = (dist[cur] + 1, order, cur, vid, outer)
            if best is None or cand < best:
                best = cand
        for di, dj in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            ni, nj = ci + di, cj + dj
            nxt = (ni, nj)
            if 0 <= ni < ph and 0 <= nj < pw and nxt not in blocked \
                    and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return best, dist


def build_geometry(links, masked: np.ndarray, ph: int, pw: int):
    """Resolve link endpoints into deterministic cell paths and edges.

    Returns ``(cells, edges, paths, used_boundary)`` where ``paths`` maps
    link index to a list of ``(i, j)`` plaquette coordinates plus a
    trailing virtual id for boundary links.
    """
    blocked = blocked_plaquettes(masked)
    cells: set[tuple[int, int]] = set()
    edges: set[tuple] = set()
    paths: dict[int, list] = {}
    used_boundary: dict[int, str] = {}
    for idx, (a, b) in enumerate(links):
        ta = parse_node(a)
        if ta[0] == "cell" and b.startswith("cell:"):
            tb = parse_node(b)
            src = (ta[1], ta[2])
            dst = (tb[1], tb[2])
            path = _bfs_cell_path(src, dst, blocked, ph, pw)
            if path is None:
                raise ValueError(f"支切无法避开遮挡: {a} <-> {b}")
            for step in path:
                cells.add(step)
            for u, v in zip(path[:-1], path[1:]):
                edges |= _blocked_edges_for_move(u, v)
            paths[idx] = path
        else:
            src = (ta[1], ta[2]) if ta[0] == "cell" else None
            if src is None:
                tb = parse_node(b)
                src = (tb[1], tb[2])
            best, _ = _nearest_boundary(src, blocked, ph, pw)
            if best is None:
                raise ValueError(f"残差无法连到外边界: {a} <-> {b}")
            cost, _order, end_plq, vid, outer = best
            path = _bfs_cell_path(src, end_plq, blocked, ph, pw)
            for step in path:
                cells.add(step)
            for u, v in zip(path[:-1], path[1:]):
                edges |= _blocked_edges_for_move(u, v)
            edges.add(outer)
            paths[idx] = path + [vid]
            used_boundary[idx] = vid
    return cells, edges, paths, used_boundary


def _pair_cost(r1: Residue, r2: Residue, ph: int, pw: int) -> int:
    return abs(r1.i - r2.i) + abs(r1.j - r2.j)


def _border_cost(r: Residue, ph: int, pw: int) -> int:
    return boundary_distance(r.i, r.j, ph, pw) + 1


def enumerate_pairings(residues: list[Residue], ph: int, pw: int,
                       limit: int = 3, forbidden: Optional[set] = None,
                       locked: Optional[list] = None):
    """Enumerate lowest-cost residue networks.

    Every residue is either paired with an opposite-charge residue or
    connected to the boundary (unbalanced).  Returns up to ``limit``
    candidates as ``(total_cost, links, pairing)`` with ``link`` items
    of the form ``(node_a, node_b, cost)``.
    """
    forbidden = forbidden or set()
    locked = locked or []
    rids = [r.rid for r in residues]
    by_id = {r.rid: r for r in residues}
    used = set()
    fixed_links: list[tuple[str, str, int]] = []
    for a, b in locked:
        r = by_id[a]
        if b.startswith("br"):
            fixed_links.append((f"cell:{r.i}_{r.j}", "boundary",
                                _border_cost(r, ph, pw)))
        else:
            s = by_id[b]
            fixed_links.append((f"cell:{r.i}_{r.j}",
                                f"cell:{s.i}_{s.j}",
                                _pair_cost(r, s, ph, pw)))
        used.add(a)
        if not b.startswith("br"):
            used.add(b)

    free = [r for r in residues if r.rid not in used]
    best: list[tuple[int, list]] = []

    def visit(pending, links):
        if len(best) >= limit * 200:
            return
        if not pending:
            total = sum(c for _, _, c in fixed_links) + sum(c for _, _, c in links)
            best.append((total, list(links)))
            return
        r = pending[0]
        rest = pending[1:]
        rid = f"cell:{r.i}_{r.j}"
        visit(rest, links + [(rid, "boundary", _border_cost(r, ph, pw))])
        for k, s in enumerate(rest):
            if s.charge == r.charge:
                continue
            key = tuple(sorted((r.rid, s.rid)))
            if key in forbidden:
                continue
            sid = f"cell:{s.i}_{s.j}"
            cost = _pair_cost(r, s, ph, pw)
            visit(rest[:k] + rest[k + 1:],
                  links + [(rid, sid, cost)])

    visit(free, [])
    best.sort(key=lambda x: x[0])
    out = []
    seen = set()
    for total, links in best:
        all_links = fixed_links + links
        sig = tuple(sorted((a, b) for a, b, _ in all_links))
        if sig in seen:
            continue
        seen.add(sig)
        out.append((total, all_links, [a for a, _, _ in all_links]))
        if len(out) >= limit:
            break
    return out


@dataclass
class UnwrapResult:
    unwrapped: np.ndarray
    components: np.ndarray
    reachable_components: list[int]
    unreachable: list[dict] = field(default_factory=list)
    start: tuple[int, int] = (0, 0)


def cut_edges_from_links(links, masked: np.ndarray, ph: int, pw: int):
    _, edges, _, _ = build_geometry(links, masked, ph, pw)
    return edges


def unwrap(wrapped: np.ndarray, masked: np.ndarray, blocked_edges: set,
           start: tuple[int, int] = (0, 0)) -> UnwrapResult:
    """Quality-guided-ish flood fill over pixels.

    The seed component keeps the wrapped value of the start pixel (its
    own integer-period offset is zero by definition).  Every other
    connected component gets an independent cycle offset, anchored to
    the first pixel that flood fill reaches inside it: components are
    never forced to share a global offset or global minimum.

    Unreachable (separate) components are still reported with their
    pixel counts and per-component anchor, clearly marked unknown.
    """
    from collections import deque

    w = np.asarray(wrapped, dtype=np.float64)
    h, width = w.shape
    m = np.asarray(masked, dtype=bool)
    unwrapped = np.full_like(w, np.nan)
    comp = np.full((h, width), -1, dtype=np.int32)

    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def blocked(a, b):
        key = _edge_key(a[0], a[1], b[0], b[1])
        return key in blocked_edges

    def flood(seed, cid, anchor_to_seed_value: bool, anchor_pixel=None):
        q = deque([seed])
        comp[seed] = cid
        anchor = seed if anchor_pixel is None else anchor_pixel
        if anchor_to_seed_value:
            unwrapped[seed] = w[seed]
        else:
            ar, ac = anchor
            unwrapped[seed] = w[seed] + unwrapped[ar, ac] - w[ar, ac]
        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < width):
                    continue
                if m[nr, nc] or comp[nr, nc] != -1:
                    continue
                if blocked((r, c), (nr, nc)):
                    continue
                d = _wrap_diff(w[nr, nc], w[r, c])
                unwrapped[nr, nc] = unwrapped[r, c] + d
                comp[nr, nc] = cid
                q.append((nr, nc))

    sr, sc = start
    if m[sr, sc]:
        valid = np.argwhere(~m)
        if len(valid) == 0:
            raise ValueError("全部像素均被遮挡")
        sr, sc = (int(valid[0][0]), int(valid[0][1]))
        start = (sr, sc)

    cid = 0
    flood((sr, sc), cid, anchor_to_seed_value=True)
    reachable = [0]
    unreachable = []
    while True:
        remaining = np.argwhere((comp == -1) & ~m)
        if len(remaining) == 0:
            break
        cid += 1
        seed = (int(remaining[0][0]), int(remaining[0][1]))
        flood(seed, cid, anchor_to_seed_value=True)
        count = int(np.sum(comp == cid))
        unreachable.append({
            "component": cid,
            "pixels": count,
            "anchor": [seed[0], seed[1]],
            "anchor_wrapped": float(w[seed]),
            "note": "整周期偏置未知：仅确定本块内部相对相位",
        })
    return UnwrapResult(unwrapped, comp, reachable, unreachable, start)
