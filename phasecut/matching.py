"""Residue pairing candidates and branch-cut path geometry.

A cut edge joins two residue nodes (plaquette centres) or a residue node to
the outer boundary (node id ``"boundary"``).  The cut is realised as a path
on the dual (plaquette) graph; every step blocks the pixel edge shared by
the two plaquettes.  Costs are Manhattan distances in plaquette steps.
"""
from __future__ import annotations

BOUNDARY = "boundary"


def boundary_cost(plaq, shape):
    """Steps from plaquette (pr, pc) to the outer boundary (min, ordered)."""
    h, w = shape
    pr, pc = plaq
    return min(pc + 1, pr + 1, (w - 1) - pc, (h - 1) - pr)


def _pair_cost(a, b):
    return abs(a["r"] - b["r"]) + abs(a["c"] - b["c"])


def _edge_key(a_id, b_id):
    return tuple(sorted((a_id, b_id)))


def _allowed(a_id, b_id, forbids):
    return _edge_key(a_id, b_id) not in forbids


def _greedy(pos, neg, forbids):
    pairs, used = [], set()
    options = []
    for p in pos:
        for n in neg:
            if _allowed(p["id"], n["id"], forbids):
                options.append((_pair_cost(p, n), p["id"], n["id"]))
    options.sort()
    by_id = {r["id"]: r for r in pos + neg}
    for cost, pid, nid in options:
        if pid in used or nid in used:
            continue
        used.add(pid)
        used.add(nid)
        pairs.append((by_id[pid], by_id[nid], cost))
    return pairs, used


def _optimal(pos, neg, shape, forbids):
    best = {"cost": None, "pairs": None}

    def rec(i, used, pairs, cost):
        if best["cost"] is not None and cost >= best["cost"]:
            return
        if i == len(pos):
            total = cost + sum(boundary_cost((n["r"], n["c"]), shape)
                               for n in neg if n["id"] not in used)
            if best["cost"] is None or total < best["cost"]:
                best["cost"] = total
                best["pairs"] = list(pairs)
            return
        p = pos[i]
        rec(i + 1, used, pairs, cost + boundary_cost((p["r"], p["c"]), shape))
        for n in neg:
            if n["id"] in used or not _allowed(p["id"], n["id"], forbids):
                continue
            used.add(n["id"])
            pairs.append((p, n, _pair_cost(p, n)))
            rec(i + 1, used, pairs, cost + pairs[-1][2])
            pairs.pop()
            used.discard(n["id"])

    rec(0, set(), [], 0)
    return best["pairs"]


def _edges_from_pairs(pairs, leftover, shape, forbids):
    edges = []
    for p, n, cost in pairs:
        edges.append({"a": p["id"], "b": n["id"], "cost": cost})
    for r in leftover:
        if not _allowed(r["id"], BOUNDARY, forbids):
            continue  # forbidden boundary link; leave to manual edit
        edges.append({"a": r["id"], "b": BOUNDARY,
                      "cost": boundary_cost((r["r"], r["c"]), shape)})
    return edges


def generate_candidates(residues, shape, locks=(), forbids=()):
    """Return up to 3 distinct pairing networks with total costs.

    ``locks`` / ``forbids`` are iterables of residue-id pairs (``"boundary"``
    allowed).  Locked pairs are forced into every candidate; forbidden pairs
    are excluded from automatic generation.
    """
    locks = {_edge_key(*k) for k in locks}
    forbids = {_edge_key(*k) for k in forbids}
    by_id = {r["id"]: r for r in residues}
    locked_pairs = []
    locked_ids = set()
    for key in locks:
        a, b = (by_id.get(key[0]), by_id.get(key[1]))
        if a is None or b is None:
            continue
        locked_pairs.append((a, b, _pair_cost(a, b)))
        locked_ids.update(key)

    pos = [r for r in residues
           if r["charge"] > 0 and r["id"] not in locked_ids]
    neg = [r for r in residues
           if r["charge"] < 0 and r["id"] not in locked_ids]

    def finish(pairs, used):
        pairs = locked_pairs + list(pairs)
        used = set(used) | locked_ids
        leftover = [r for r in residues if r["id"] not in used]
        edges = _edges_from_pairs(pairs, leftover, shape, forbids)
        return {"edges": edges, "total_cost": sum(e["cost"] for e in edges)}

    candidates = []
    g_pairs, g_used = _greedy(pos, neg, forbids)
    cand = dict(finish(g_pairs, g_used)); cand["name"] = "greedy"
    candidates.append(cand)

    o_pairs = _optimal(pos, neg, shape, forbids)
    o_used = {r["id"] for pair in o_pairs for r in pair[:2]}
    cand = dict(finish(o_pairs, o_used)); cand["name"] = "optimal"
    candidates.append(cand)

    # Boundary: locked pairs kept, everything else to the outer boundary.
    cand = dict(finish([], set()))
    cand["name"] = "boundary"
    candidates.append(cand)

    return candidates
    return unique


def cut_path(a_plaq, b_plaq):
    """Deterministic L-shaped plaquette path (horizontal, then vertical)."""
    path = [tuple(a_plaq)]
    r, c = a_plaq
    while c != b_plaq[1]:
        c += 1 if b_plaq[1] > c else -1
        path.append((r, c))
    while r != b_plaq[0]:
        r += 1 if b_plaq[0] > r else -1
        path.append((r, c))
    return path


def boundary_path(plaq, shape):
    """Plaquette path from ``plaq`` to just outside the nearest boundary."""
    h, w = shape
    pr, pc = plaq
    options = [(pc + 1, (0, -1)), (pr + 1, (-1, 0)),
               ((w - 1) - pc, (0, 1)), ((h - 1) - pr, (1, 0))]
    _, (dr, dc) = min(options, key=lambda t: t[0])
    path = [(pr, pc)]
    r, c = pr, pc
    while 0 <= r < h - 1 and 0 <= c < w - 1:
        r, c = r + dr, c + dc
        path.append((r, c))
    return path


def plaquette_step_to_pixel_edge(p, q):
    """Pixel edge crossed when stepping between adjacent plaquettes p -> q."""
    (r1, c1), (r2, c2) = p, q
    if r1 == r2:
        c = max(c1, c2)
        return ((r1, c), (r1 + 1, c))
    r = max(r1, r2)
    return ((r, c1), (r, c1 + 1))


def path_to_pixel_edges(path):
    return [plaquette_step_to_pixel_edge(p, q)
            for p, q in zip(path, path[1:])]
