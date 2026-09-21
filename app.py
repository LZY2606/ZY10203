"""相位支切室 — FastAPI service.

Serves the operation page and the JSON API linking wrapped phase, quality,
residue charges, branch cuts and unwrapped results.  State (cut versions and
unwrap runs) lives in SQLite so runs are traceable, exportable and
replayable.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from phasecut import core, db, fixture, matching

DATASET = fixture.build_fixture()
RESIDUES = core.residue_list(DATASET["wrapped"], DATASET["valid"])
RESIDUE_BY_ID = {r["id"]: r for r in RESIDUES}
SHAPE = tuple(DATASET["shape"])

app = FastAPI(title="相位支切室")


def _grid(arr):
    return [[None if (isinstance(v, float) and math.isnan(v)) else v
             for v in row] for row in np.asarray(arr).tolist()]


def _resolve_edges(edges):
    """Turn stored residue-id edges into pixel-edge segments.

    Raises HTTPException 400 if a cut would cross an occluded pixel.
    """
    valid = DATASET["valid"]
    segments = []
    for edge in edges:
        a_id, b_id = edge.get("a"), edge.get("b")
        if a_id not in RESIDUE_BY_ID:
            raise HTTPException(400, f"unknown residue id: {a_id}")
        a = RESIDUE_BY_ID[a_id]
        if b_id == matching.BOUNDARY:
            path = matching.boundary_path((a["r"], a["c"]), SHAPE)
        elif b_id in RESIDUE_BY_ID:
            b = RESIDUE_BY_ID[b_id]
            path = matching.cut_path((a["r"], a["c"]), (b["r"], b["c"]))
        else:
            raise HTTPException(400, f"unknown residue id: {b_id}")
        for (p, q) in matching.path_to_pixel_edges(path):
            if not (valid[p] and valid[q]):
                raise HTTPException(
                    400, f"cut {a_id}->{b_id} crosses occluded pixel"
                         f" at edge {p}-{q}")
            segments.append([list(p), list(q)])
    return segments


def _run_unwrap(cut_version, start):
    segments = _resolve_edges(cut_version["edges"])
    blocked = [(tuple(s[0]), tuple(s[1])) for s in segments]
    unwrapped, labels, components = core.unwrap(
        DATASET["wrapped"], DATASET["valid"], blocked, start)
    result = {
        "components": components,
        "unreachable": [c["id"] for c in components
                        if not c["reachable_from_start"]],
        "unwrapped": _grid(unwrapped),
        "labels": np.asarray(labels).tolist(),
    }
    digest_payload = {
        "components": components,
        "unwrapped": [[None if v is None else round(v, 9) for v in row]
                      for row in result["unwrapped"]],
    }
    digest = hashlib.sha256(json.dumps(
        digest_payload, sort_keys=True).encode()).hexdigest()
    return result, digest


class CandidateRequest(BaseModel):
    locks: list[list[str]] = []
    forbids: list[list[str]] = []


class CutRequest(BaseModel):
    name: str = "cut"
    note: str = ""
    edges: list[dict]


class UnwrapRequest(BaseModel):
    cut_version_id: int
    start: list[int]


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/dataset")
def get_dataset():
    return {
        "shape": DATASET["shape"],
        "wrapped": _grid(DATASET["wrapped"]),
        "quality": _grid(DATASET["quality"]),
        "valid": np.asarray(DATASET["valid"]).astype(int).tolist(),
        "residues": RESIDUES,
    }


@app.post("/api/candidates")
def post_candidates(req: CandidateRequest):
    cands = matching.generate_candidates(
        RESIDUES, SHAPE, locks=req.locks, forbids=req.forbids)
    return {"residues": RESIDUES, "candidates": cands}


@app.post("/api/cuts")
def post_cut(req: CutRequest):
    segments = _resolve_edges(req.edges)  # validates occlusion first
    vid = db.save_cut_version(req.name, req.edges, req.note)
    return {"id": vid, "segments": segments}


@app.get("/api/cuts")
def list_cuts():
    return {"versions": db.list_cut_versions()}


@app.get("/api/cuts/{vid}")
def get_cut(vid: int):
    cv = db.get_cut_version(vid)
    if cv is None:
        raise HTTPException(404, "cut version not found")
    cv["segments"] = _resolve_edges(cv["edges"])
    return cv


@app.post("/api/unwrap")
def post_unwrap(req: UnwrapRequest):
    cv = db.get_cut_version(req.cut_version_id)
    if cv is None:
        raise HTTPException(404, "cut version not found")
    try:
        result, digest = _run_unwrap(cv, req.start)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    run_id = db.save_run(cv["id"], req.start, digest, result)
    return {"run_id": run_id, "digest": digest,
            "cut_version_id": cv["id"], "start": req.start, **result}


@app.get("/api/runs")
def list_runs():
    return {"runs": [{k: v for k, v in r.items() if k != "result"}
                     for r in db.list_runs()]}


@app.post("/api/runs/{rid}/replay")
def replay_run(rid: int):
    run = db.get_run(rid)
    if run is None:
        raise HTTPException(404, "run not found")
    cv = db.get_cut_version(run["cut_version_id"])
    if cv is None:
        raise HTTPException(410, "cut version of this run is missing")
    result, digest = _run_unwrap(cv, run["start"])
    return {"run_id": rid, "digest": digest,
            "stored_digest": run["digest"],
            "match": digest == run["digest"], **result}


@app.get("/api/runs/export")
def export_runs():
    return db.export_all()


@app.post("/api/runs/import")
def import_runs(payload: dict):
    try:
        return db.import_all(payload, replace=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/admin/clear")
def clear_db():
    db.clear_all()
    return {"cleared": True}
