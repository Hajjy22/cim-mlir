"""Differential-style check on `cim-bench amortize`'s actual output.

Not a comparison against an independent implementation of the formula (the
formula itself is unit-tested against hand-worked numbers in
test/unit/cost_report_test.cpp) -- this is the other half: run the real
shipped binary against the two real target files this feature exists to
compare, and check the property the whole feature is for, on real output:
that a non-volatile and a volatile target produce genuinely different curve
shapes, not just different numbers on the same shape (docs/roadmap.md's M3
section, bench/plots/plot_amortization.py).
"""

import json

from conftest import run


def amortize(cim_bench, target, tmp_path):
    out = tmp_path / f"{target}.json"
    result = run([cim_bench, "amortize", "--target", target, "--out", str(out)])
    assert result.returncode == 0, result.stderr
    with open(out) as fh:
        return json.load(fh)


def test_nonvolatile_target_has_no_leakage_floor(cim_bench, tmp_path):
    data = amortize(cim_bench, "erbium-8t", tmp_path)
    assert data["persistent"] is True
    assert data["standby_leakage_uw_per_tile"] == 0.0

    values = [p["amortized_install_pj_per_inference"] for p in data["points"]]
    # Strictly decreasing all the way out -- nothing floors it.
    assert all(a > b for a, b in zip(values, values[1:])), values
    # cim-bench.cpp's kSweep steps by 10x; amortizing over 10x more
    # inferences should cut the cost by very close to 10x -- no floor
    # eating into it.
    ratio = values[-2] / values[-1]
    assert 9 < ratio < 11, f"expected ~10x, got {ratio}x ({values[-2:]})"


def test_volatile_target_hits_a_nonzero_floor(cim_bench, tmp_path):
    data = amortize(cim_bench, "generic-digital-cim", tmp_path)
    assert data["persistent"] is False
    assert data["standby_leakage_uw_per_tile"] > 0.0

    values = [p["amortized_install_pj_per_inference"] for p in data["points"]]
    # Non-increasing (amortizing never makes it worse)...
    assert all(a >= b for a, b in zip(values, values[1:])), values
    # ...but the last two swept points (1e8 and 1e9 inferences) are within
    # 1% of each other: the curve has visibly flattened, unlike the
    # non-volatile one, which is still shrinking ~1000x per decade at the
    # same point in the sweep.
    ratio = values[-2] / values[-1]
    assert ratio < 1.01, f"expected the curve to have flattened, got {ratio}x"


def test_persistence_changes_which_target_is_better_not_just_by_how_much(
    cim_bench, tmp_path
):
    """The actual claim: the two curves cross.

    At few inferences the volatile target's cheaper install cost (no
    non-volatile write penalty) wins; at many inferences the non-volatile
    target's install cost has amortized away while the volatile target is
    stuck paying standby leakage forever, so it wins instead. A reader
    comparing the two targets at a single N would see only a magnitude
    difference and could not tell this crossover exists at all.
    """
    nonvolatile = amortize(cim_bench, "erbium-8t", tmp_path)
    volatile = amortize(cim_bench, "generic-digital-cim", tmp_path)

    nv_by_n = {p["inferences"]: p["amortized_install_pj_per_inference"]
               for p in nonvolatile["points"]}
    v_by_n = {p["inferences"]: p["amortized_install_pj_per_inference"]
              for p in volatile["points"]}
    shared_ns = sorted(set(nv_by_n) & set(v_by_n))
    assert shared_ns, "the two sweeps must share at least one inference count"

    nonvolatile_wins_somewhere = any(nv_by_n[n] < v_by_n[n] for n in shared_ns)
    volatile_wins_somewhere = any(v_by_n[n] < nv_by_n[n] for n in shared_ns)
    assert nonvolatile_wins_somewhere and volatile_wins_somewhere, (
        "expected the curves to cross -- one target should win at low "
        "inference counts and the other at high ones, not the same one "
        f"everywhere: nonvolatile={nv_by_n}, volatile={v_by_n}"
    )
