import math

import numpy as np
import pytest

from phasecut import core, fixture, matching


@pytest.fixture()
def ds():
    return fixture.build_fixture()


# ---------- 半开区间规约 ----------

def test_wrap_half_open_endpoints_exact():
    # 恰好等于 ±π 的差值规约到半开区间 (-π, π] 的同一端点，不翻转。
    assert core.wrap_phase(math.pi) == math.pi
    assert core.wrap_phase(-math.pi) == math.pi
    assert core.wrap_phase(3 * math.pi) == math.pi
    out = core.wrap_phase(np.array([math.pi, -math.pi]))
    assert out.tolist() == [math.pi, math.pi]


def test_wrap_endpoint_neighbours_do_not_flip():
    # ±π 的相邻浮点数也保持各自一侧，不受舍入影响。
    below_pi = math.nextafter(math.pi, 0.0)
    above_pi = math.nextafter(math.pi, math.inf)
    below_mpi = math.nextafter(-math.pi, 0.0)
    above_mpi = math.nextafter(-math.pi, -math.inf)
    assert 0 < core.wrap_phase(below_pi) < math.pi
    assert -math.pi < core.wrap_phase(above_pi) < 0
    assert -math.pi < core.wrap_phase(below_mpi) < 0
    assert 0 < core.wrap_phase(above_mpi) < math.pi


def test_wrap_idempotent_and_range():
    x = np.linspace(-20.0, 20.0, 40001)
    w = core.wrap_phase(x)
    assert np.all(w > -math.pi) and np.all(w <= math.pi)
    assert np.array_equal(core.wrap_phase(w), w)


def test_residue_charge_stable_at_half_period():
    # 环路差值恰好落在 ±π 上时，电荷符号由半开区间规则唯一决定。
    wrapped = np.array([[0.0, math.pi], [0.0, math.pi]])
    valid = np.ones((2, 2), dtype=bool)
    charges, computed = core.detect_residues(wrapped, valid)
    assert computed[0, 0]
    assert charges[0, 0] == 1  # (+π, 0, +π, 0) 求和为 +2π


# ---------- 固定 fixture ----------

def test_fixture_residues(ds):
    residues = core.residue_list(ds["wrapped"], ds["valid"])
    got = {(r["r"], r["c"]): r["charge"] for r in residues}
    assert got == {(4, 3): 1, (10, 5): -1, (7, 15): 1}


def test_fixture_occlusion_splits_two_blocks(ds):
    unwrapped, labels, components = core.unwrap(
        ds["wrapped"], ds["valid"], [], (0, 0))
    assert len(components) == 2
    assert not np.any(ds["valid"][:, 10])
    assert labels[0, 0] != labels[0, 19]


def test_occluded_plaquettes_carry_no_charge(ds):
    charges, computed = core.detect_residues(ds["wrapped"], ds["valid"])
    assert not np.any(computed[:, 9])   # 跨越遮挡列的环路不参与
    assert not np.any(computed[:, 10])


# ---------- 自动配对候选 ----------

def test_candidates_multiple_with_costs(ds):
    residues = core.residue_list(ds["wrapped"], ds["valid"])
    cands = matching.generate_candidates(residues, tuple(ds["shape"]))
    assert len(cands) >= 2
    by_name = {c["name"]: c for c in cands}
    for cand in cands:
        assert cand["total_cost"] == sum(e["cost"] for e in cand["edges"])
    assert by_name["optimal"]["total_cost"] <= by_name["greedy"]["total_cost"]
    # 最优解：正负残差配对 + 未平衡残差连外边界。
    opt_edges = {tuple(sorted((e["a"], e["b"]))) for e in by_name["optimal"]["edges"]}
    assert ("r0", "r2") in opt_edges          # 左块正负残差配对
    assert ("boundary", "r1") in opt_edges    # 右块未平衡残差连外边界
    bnd = by_name["boundary"]
    assert all(e["b"] == "boundary" for e in bnd["edges"])


def test_candidates_respect_locks_and_forbids(ds):
    residues = core.residue_list(ds["wrapped"], ds["valid"])
    cands = matching.generate_candidates(
        residues, tuple(ds["shape"]), forbids=[("r0", "r2")])
    for cand in cands:
        assert ("r0", "r2") not in {
            tuple(sorted((e["a"], e["b"]))) for e in cand["edges"]}
    locked = matching.generate_candidates(
        residues, tuple(ds["shape"]), locks=[("r0", "r1")])
    for cand in locked:
        assert ("r0", "r1") in {
            tuple(sorted((e["a"], e["b"]))) for e in cand["edges"]}


# ---------- 支切几何与遮挡 ----------

def test_cut_path_blocks_expected_pixel_edges():
    path = matching.cut_path((4, 3), (10, 6))
    edges = matching.path_to_pixel_edges(path)
    assert edges[0] == ((4, 4), (5, 4))
    assert ((10, 6), (10, 7)) in edges or ((10, 6), (11, 6)) in edges


def test_boundary_path_reaches_edge(ds):
    path = matching.boundary_path((7, 15), tuple(ds["shape"]))
    assert path[-1][1] == ds["shape"][1] - 1  # 步出最右列的环路网


# ---------- 展开与连通域偏置 ----------

def _optimal_segments(ds):
    residues = core.residue_list(ds["wrapped"], ds["valid"])
    cands = matching.generate_candidates(residues, tuple(ds["shape"]))
    opt = next(c for c in cands if c["name"] == "optimal")
    by_id = {r["id"]: r for r in residues}
    segments = []
    for e in opt["edges"]:
        a = by_id[e["a"]]
        if e["b"] == "boundary":
            path = matching.boundary_path((a["r"], a["c"]), tuple(ds["shape"]))
        else:
            b = by_id[e["b"]]
            path = matching.cut_path((a["r"], a["c"]), (b["r"], b["c"]))
        segments += matching.path_to_pixel_edges(path)
    return segments


def test_unwrap_internal_consistency_with_cuts(ds):
    segments = _optimal_segments(ds)
    unwrapped, labels, components = core.unwrap(
        ds["wrapped"], ds["valid"], segments, (0, 0))
    blocked = {core.norm_edge(*map(tuple, s)) for s in segments}
    h, w = ds["shape"]
    for r in range(h):
        for c in range(w):
            if labels[r, c] < 0:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr >= h or nc >= w or labels[nr, nc] != labels[r, c]:
                    continue
                if core.norm_edge((r, c), (nr, nc)) in blocked:
                    continue
                got = core.wrap_phase(unwrapped[nr, nc] - unwrapped[r, c])
                want = core.wrap_phase(ds["wrapped"][nr, nc]
                                       - ds["wrapped"][r, c])
                assert abs(got - want) < 1e-9


def test_each_component_determines_only_its_own_offset(ds):
    segments = _optimal_segments(ds)
    u_left, labels_left, comps_left = core.unwrap(
        ds["wrapped"], ds["valid"], segments, (0, 0))
    u_right, labels_right, comps_right = core.unwrap(
        ds["wrapped"], ds["valid"], segments, (15, 0))
    assert len(comps_left) == len(comps_right) == 2
    # 起始点所在连通域可达，另一连通域标记为未可达。
    reach_left = [c for c in comps_left if c["reachable_from_start"]]
    assert len(reach_left) == 1 and comps_left[1]["reachable_from_start"] is False
    # 同一连通域内，不同起点只相差各自的一个整数 2π 偏置。
    for r in range(ds["shape"][0]):
        for c in range(0, 9):
            if not ds["valid"][r, c]:
                continue
            k = (u_left[r, c] - u_right[r, c]) / core.TWO_PI
            assert abs(k - round(k)) < 1e-9
    ks = [round((u_left[r, c] - u_right[r, c]) / core.TWO_PI)
          for r in range(ds["shape"][0]) for c in range(0, 9)
          if ds["valid"][r, c]]
    assert len(set(ks)) == 1
    # 不做全局最小值假定：两个连通域各自锚定在自身种子的包裹值。
    for comps, u in ((comps_left, u_left), (comps_right, u_right)):
        for comp in comps:
            sr, sc = comp["seed"]
            assert u[sr, sc] == ds["wrapped"][sr, sc]
