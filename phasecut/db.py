"""SQLite persistence for cut versions and unwrap runs."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_PATH = os.path.join(os.getcwd(), "phasecut.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cut_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    edges_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cut_version_id INTEGER NOT NULL,
    start_r INTEGER NOT NULL,
    start_c INTEGER NOT NULL,
    digest TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def db_path():
    return os.environ.get("PHASECUT_DB", DEFAULT_PATH)


def connect():
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_cut_version(name, edges, note=""):
    conn = connect()
    cur = conn.execute(
        "INSERT INTO cut_versions (name, note, edges_json, created_at)"
        " VALUES (?, ?, ?, ?)",
        (name, note, json.dumps(edges), _now()))
    conn.commit()
    vid = cur.lastrowid
    conn.close()
    return vid


def list_cut_versions():
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM cut_versions ORDER BY id").fetchall()
    conn.close()
    return [_cut_row(r) for r in rows]


def get_cut_version(vid):
    conn = connect()
    row = conn.execute(
        "SELECT * FROM cut_versions WHERE id = ?", (vid,)).fetchone()
    conn.close()
    return _cut_row(row) if row else None


def _cut_row(row):
    return {"id": row["id"], "name": row["name"], "note": row["note"],
            "edges": json.loads(row["edges_json"]),
            "created_at": row["created_at"]}


def save_run(cut_version_id, start, digest, result):
    conn = connect()
    cur = conn.execute(
        "INSERT INTO runs (cut_version_id, start_r, start_c, digest,"
        " result_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (cut_version_id, int(start[0]), int(start[1]), digest,
         json.dumps(result), _now()))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def list_runs():
    conn = connect()
    rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    conn.close()
    return [_run_row(r) for r in rows]


def get_run(rid):
    conn = connect()
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (rid,)).fetchone()
    conn.close()
    return _run_row(row) if row else None


def _run_row(row):
    return {"id": row["id"], "cut_version_id": row["cut_version_id"],
            "start": [row["start_r"], row["start_c"]],
            "digest": row["digest"], "result": json.loads(row["result_json"]),
            "created_at": row["created_at"]}


def export_all():
    return {"format": "phasecut-export/1",
            "cut_versions": list_cut_versions(), "runs": list_runs()}


def import_all(payload, replace=False):
    if payload.get("format") != "phasecut-export/1":
        raise ValueError("unrecognised export format")
    conn = connect()
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    n_cuts = n_runs = 0
    for cv in payload.get("cut_versions", []):
        conn.execute(
            f"{verb} INTO cut_versions (id, name, note, edges_json,"
            " created_at) VALUES (?, ?, ?, ?, ?)",
            (cv["id"], cv["name"], cv.get("note", ""),
             json.dumps(cv["edges"]), cv["created_at"]))
        n_cuts += 1
    for run in payload.get("runs", []):
        conn.execute(
            f"{verb} INTO runs (id, cut_version_id, start_r, start_c,"
            " digest, result_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run["id"], run["cut_version_id"], run["start"][0],
             run["start"][1], run["digest"], json.dumps(run["result"]),
             run["created_at"]))
        n_runs += 1
    conn.commit()
    conn.close()
    return {"cut_versions": n_cuts, "runs": n_runs}


def clear_all():
    conn = connect()
    conn.execute("DELETE FROM runs")
    conn.execute("DELETE FROM cut_versions")
    conn.commit()
    conn.close()
