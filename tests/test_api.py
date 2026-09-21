import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASECUT_DB", str(tmp_path / "test.db"))
    import app
    return TestClient(app.app)


def _optimal_edges(client):
    j = client.post("/api/candidates", json={}).json()
    opt = next(c for c in j["candidates"] if c["name"] == "optimal")
    return [{"a": e["a"], "b": e["b"]} for e in opt["edges"]]


def test_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "相位支切室" in r.text


def test_dataset_endpoint(client):
    j = client.get("/api/dataset").json()
    assert j["shape"] == [16, 20]
    assert {(r["r"], r["c"]): r["charge"] for r in j["residues"]} == {
        (4, 3): 1, (10, 5): -1, (7, 15): 1}


def test_cut_crossing_occlusion_rejected(client):
    # r0(4,3) 直连 r1(7,15) 的 L 形路径必穿过被遮挡的第 10 列。
    r = client.post("/api/cuts", json={
        "name": "bad", "edges": [{"a": "r0", "b": "r1"}]})
    assert r.status_code == 400
    assert "occluded" in r.json()["detail"]


def test_unwrap_run_components_and_replay_after_edit(client):
    vid = client.post("/api/cuts", json={
        "name": "v1", "edges": _optimal_edges(client)}).json()["id"]
    j = client.post("/api/unwrap", json={
        "cut_version_id": vid, "start": [0, 0]}).json()
    assert len(j["components"]) == 2
    assert j["unreachable"] == [1]
    digest_v1 = j["digest"]

    # 编辑支切（换用 frugal 候选）后重放旧运行，结果必须可追溯一致。
    alt = next(c for c in client.post("/api/candidates", json={}).json()
                  ["candidates"] if c["name"] == "boundary")
    client.post("/api/cuts", json={
        "name": "v2", "edges": [{"a": e["a"], "b": e["b"]}
                                for e in alt["edges"]]})
    rep = client.post(f"/api/runs/{j['run_id']}/replay").json()
    assert rep["match"] is True
    assert rep["digest"] == digest_v1


def test_export_clear_import_replay(client):
    vid = client.post("/api/cuts", json={
        "name": "v1", "edges": _optimal_edges(client)}).json()["id"]
    run = client.post("/api/unwrap", json={
        "cut_version_id": vid, "start": [0, 0]}).json()
    exported = client.get("/api/runs/export").json()
    assert exported["format"] == "phasecut-export/1"
    assert len(exported["runs"]) == 1

    client.post("/api/admin/clear")
    assert client.get("/api/runs").json()["runs"] == []

    counts = client.post("/api/runs/import", json=exported).json()
    assert counts == {"cut_versions": 1, "runs": 1}
    rep = client.post(f"/api/runs/{run['run_id']}/replay").json()
    assert rep["match"] is True
    assert rep["digest"] == run["digest"]


def test_start_on_occluded_pixel_rejected(client):
    vid = client.post("/api/cuts", json={
        "name": "v1", "edges": _optimal_edges(client)}).json()["id"]
    r = client.post("/api/unwrap", json={
        "cut_version_id": vid, "start": [0, 10]})
    assert r.status_code == 400
