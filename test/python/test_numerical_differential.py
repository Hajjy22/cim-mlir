"""Differential test: the compiled pipeline vs numpy.

`test/mlir/pipeline_e2e_test.cpp` already checks the compiler's arithmetic
against a C++ triple loop, in-process, on nine hand-picked cases. This is
the other half of that argument, and it is deliberately different on every
axis:

  * A different reference. numpy's integer matmul was written by people who
    have never seen this repository. The C++ reference and cim-partition
    were written by the same author from the same mental model of what
    `matmul_transpose_b` means -- if that model is wrong, they agree with
    each other and both are wrong. numpy will not.
  * A different path to the answer. This drives the shipped `cim-opt` and
    `cim-run` binaries as subprocesses over textual IR, so it covers the
    parse/print round trip, the CLI plumbing and the pass-pipeline string
    that the in-process test bypasses entirely.
  * Different, and many more, inputs. Randomized shapes and contents rather
    than nine fixed cases, so tiling arithmetic gets swept instead of
    sampled.

Plain seeded `random`, not hypothesis: each case costs two subprocesses, so
hypothesis's per-example deadline fights the test, and its shrinking works
on the byte level rather than on the axis that matters. The shrink that
actually helps here -- keep failing while halving N and K -- is the twenty
lines at the bottom of this file.
"""

import os
import random
import subprocess

import numpy as np
import pytest

from conftest import REPO_ROOT, run

TINY_TARGET = os.path.join("test", "targets", "tiny-4x4.yaml")

# tiny-4x4.yaml: 2 tiles of 4x4. Shapes are multiples of 4 so a block covers
# a tile exactly; the padding path is a separate concern and is not what
# this test is for. Anything past N=8, K=8 reprograms tiles mid-execution,
# which is where tiling bugs live.
TILE = 4

SEED = int(os.environ.get("CIM_TEST_SEED", "0x5eed1234abcd"), 0)


def build_module(w, act):
    """Emit a self-contained module computing act @ w.T.

    Mirrors buildModule() in test/mlir/pipeline_e2e_test.cpp on purpose: the
    two tests must feed the compiler the same shape of input, or a
    disagreement between them would be about the IR rather than about the
    arithmetic. In particular the activation is staged through a
    memref.alloc, because cim-detect requires exactly one constant operand.
    """
    n, k = w.shape
    rows = ", ".join("[" + ", ".join(str(int(v)) for v in row) + "]" for row in w)
    acts = ", ".join(str(int(v)) for v in act)
    return f"""\
memref.global "private" constant @w : memref<{n}x{k}xi8> = dense<[{rows}]>
memref.global "private" constant @a : memref<1x{k}xi8> = dense<[[{acts}]]>
func.func private @cim_print_i32(memref<*xi32>)
func.func @main() {{
  %w = memref.get_global @w : memref<{n}x{k}xi8>
  %aInit = memref.get_global @a : memref<1x{k}xi8>
  %a = memref.alloc() : memref<1x{k}xi8>
  memref.copy %aInit, %a : memref<1x{k}xi8> to memref<1x{k}xi8>
  %out = memref.alloc() : memref<1x{n}xi32>
  linalg.matmul_transpose_b ins(%a, %w : memref<1x{k}xi8>, memref<{n}x{k}xi8>)
    outs(%out : memref<1x{n}xi32>)
  %u = memref.cast %out : memref<1x{n}xi32> to memref<*xi32>
  func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()
  memref.dealloc %a : memref<1x{k}xi8>
  memref.dealloc %out : memref<1x{n}xi32>
  return
}}
"""


class PipelineError(Exception):
    pass


def compile_and_run(cim_opt, cim_run, w, act, target=TINY_TARGET):
    """Run the real toolchain over one matmul; return (outputs, profile)."""
    source = build_module(w, act)

    compiled = run(
        [cim_opt, "-", "--cim-detect", f"--cim-partition=target-yaml={target}"],
        stdin=source)
    if compiled.returncode != 0:
        raise PipelineError(f"cim-opt failed:\n{compiled.stderr}\n{source}")

    # If the matmul was not offloaded, cim-run would fail on the leftover
    # linalg op -- but say so explicitly, because "the compiler declined"
    # and "the compiler got it wrong" are entirely different bugs and the
    # test must not report one as the other.
    if "cim.program" not in compiled.stdout:
        raise PipelineError(
            "cim-partition emitted no cim.program, so nothing was offloaded "
            f"and this case proves nothing:\n{compiled.stdout}")

    executed = run([cim_run, f"--target-yaml={target}", "--profile", "-"],
                   stdin=compiled.stdout)
    if executed.returncode != 0:
        raise PipelineError(
            f"cim-run failed:\n{executed.stderr}\n--- IR ---\n{compiled.stdout}")

    return parse_output(executed.stdout), parse_profile(executed.stdout)


def parse_output(text):
    for line in text.splitlines():
        if line.startswith("cim_print_i32"):
            data = line.split("data=[", 1)[1].rstrip("]")
            return np.array([int(t) for t in data.split(",") if t],
                            dtype=np.int64)
    raise PipelineError(f"no cim_print_i32 line in cim-run output:\n{text}")


def parse_profile(text):
    for line in text.splitlines():
        if line.startswith("cimrt_profile"):
            return {k: int(v) for k, v in
                    (tok.split("=") for tok in line.split()[1:])}
    return {}


def reference(w, act):
    """out[n] = sum_k W[n][k] * act[k] -- what matmul_transpose_b means."""
    return act.astype(np.int64) @ w.astype(np.int64).T


def check(cim_opt, cim_run, w, act, label):
    got, _ = compile_and_run(cim_opt, cim_run, w, act)
    want = reference(w, act)
    assert got.shape == want.shape, (
        f"{label}: got {got.shape} outputs, numpy says {want.shape}")
    if not np.array_equal(got, want):
        bad = np.flatnonzero(got != want)[:8]
        pytest.fail(
            f"{label}: {len(np.flatnonzero(got != want))} of {want.size} "
            f"outputs differ from numpy.\n"
            f"  indices {list(bad)}\n"
            f"  ours    {[int(got[i]) for i in bad]}\n"
            f"  numpy   {[int(want[i]) for i in bad]}\n"
            f"  W =\n{w}\n  act = {act}")


# ---------------------------------------------------------------------------
# Fixed cases: the values that break sign handling
# ---------------------------------------------------------------------------

FIXED = {
    "identity": (np.eye(4, dtype=np.int8), np.array([7, -9, 42, -128], np.int8)),
    "all-zero-weights": (np.zeros((4, 4), np.int8),
                         np.array([1, 2, 3, 4], np.int8)),
    "all-zero-activations": (np.full((4, 4), 127, np.int8),
                             np.zeros(4, np.int8)),
    # -128 has no positive counterpart; a bad int8/uint8 round trip survives
    # every all-positive test and dies here.
    "int8-extremes": (np.array([[-128, 127, -128, 127],
                                [127, -128, 127, -128],
                                [-128, -128, -128, -128],
                                [127, 127, 127, 127]], np.int8),
                      np.array([-128, 127, -128, 127], np.int8)),
    "min-times-min": (np.full((4, 4), -128, np.int8),
                      np.full(4, -128, np.int8)),
    # Non-symmetric across N and K: a transposed weight block computes the
    # right answer on every square case and the wrong one here.
    "rectangular-wide-k": (np.arange(-16, 16, dtype=np.int64)
                           .reshape(4, 8).astype(np.int8),
                           np.arange(-4, 4, dtype=np.int8)),
    "rectangular-tall-n": (np.arange(-16, 16, dtype=np.int64)
                           .reshape(8, 4).astype(np.int8),
                           np.array([3, -5, 7, -11], np.int8)),
}


@pytest.mark.parametrize("label", sorted(FIXED))
def test_fixed_case_matches_numpy(cim_opt, cim_run, label):
    w, act = FIXED[label]
    check(cim_opt, cim_run, w, act, label)


# ---------------------------------------------------------------------------
# Swept shapes: every tiling configuration the tiny target can reach
# ---------------------------------------------------------------------------

SHAPES = [(n, k) for n in (4, 8, 12, 16) for k in (4, 8, 12)]


@pytest.mark.parametrize("n,k", SHAPES, ids=[f"{n}x{k}" for n, k in SHAPES])
def test_swept_shape_matches_numpy(cim_opt, cim_run, n, k):
    """One random instance per shape.

    The shape grid is the point, not the values: N/4 x K/4 blocks over 2
    tiles covers single-block, N-tiled, K-tiled, both-tiled, and
    more-blocks-than-tiles (which forces mid-execution reprogramming) in one
    parametrized sweep.
    """
    rng = random.Random(SEED ^ (n << 8) ^ k)
    w = np.array([rng.randint(-128, 127) for _ in range(n * k)],
                 dtype=np.int8).reshape(n, k)
    act = np.array([rng.randint(-128, 127) for _ in range(k)], dtype=np.int8)
    check(cim_opt, cim_run, w, act, f"{n}x{k} random")


def test_reprogramming_a_tile_does_not_disturb_earlier_results(cim_opt,
                                                                cim_run):
    """16x4 needs 4 blocks on a 2-tile device, so tiles are reused.

    Reuse must be numerically transparent. The profile is asserted too: if
    cim-partition stopped tiling and emitted one giant program, the numbers
    would still be right and the property under test would have silently
    stopped being exercised.
    """
    rng = random.Random(SEED)
    w = np.array([rng.randint(-128, 127) for _ in range(16 * 4)],
                 dtype=np.int8).reshape(16, 4)
    act = np.array([rng.randint(-128, 127) for _ in range(4)], dtype=np.int8)

    got, profile = compile_and_run(cim_opt, cim_run, w, act)
    assert np.array_equal(got, reference(w, act))
    assert profile.get("programs", 0) >= 4, (
        f"expected at least one program per 4x4 block, got {profile}")
    assert profile.get("mvms", 0) >= 4, f"expected one mvm per block: {profile}"


# ---------------------------------------------------------------------------
# Randomized sweep with shape shrinking
# ---------------------------------------------------------------------------

CASES = int(os.environ.get("CIM_RANDOM_CASES", "24"))


def shrink(cim_opt, cim_run, w, act):
    """Drop tile blocks off N and K while the case still fails.

    Content shrinking is not attempted. A wrong answer on random int8 data
    is a structural bug -- a bad offset, a transposed block, a dropped
    partial sum -- and those reproduce at any values, so the useful thing to
    minimize is the tiling configuration. A 4x4 failure names one block; a
    16x12 failure names forty-eight and reads like noise.

    One tile at a time rather than halving: shapes here are multiples of 4,
    not powers of two, and halving 12 gives 6, which is not a whole number
    of blocks. Rejecting it leaves 12 unshrinkable -- which is exactly what
    happened the first time this was written, so the ladder is deliberate.
    """
    def fails(candidate_w, candidate_act):
        try:
            got, _ = compile_and_run(cim_opt, cim_run, candidate_w,
                                     candidate_act)
        except PipelineError:
            return True
        return not np.array_equal(got, reference(candidate_w, candidate_act))

    progress = True
    while progress:
        progress = False
        while w.shape[0] > TILE:
            candidate = w[:w.shape[0] - TILE, :]
            if not fails(candidate, act):
                break
            w, progress = candidate, True
        while w.shape[1] > TILE:
            width = w.shape[1] - TILE
            if not fails(w[:, :width], act[:width]):
                break
            w, act, progress = w[:, :width], act[:width], True
    return w, act


def test_random_matmuls_match_numpy(cim_opt, cim_run):
    rng = random.Random(SEED)

    for case in range(CASES):
        n = rng.choice([4, 8, 12, 16])
        k = rng.choice([4, 8, 12])
        w = np.array([rng.randint(-128, 127) for _ in range(n * k)],
                     dtype=np.int8).reshape(n, k)
        act = np.array([rng.randint(-128, 127) for _ in range(k)],
                       dtype=np.int8)

        try:
            got, _ = compile_and_run(cim_opt, cim_run, w, act)
            ok = np.array_equal(got, reference(w, act))
        except PipelineError as exc:
            got, ok = None, False
            failure = str(exc)
        else:
            failure = None

        if ok:
            continue

        small_w, small_act = shrink(cim_opt, cim_run, w, act)
        pytest.fail(
            f"case {case} (seed {SEED:#x}) disagreed with numpy at "
            f"{n}x{k}; smallest failing shape {small_w.shape}\n"
            f"  W =\n{small_w}\n  act = {small_act}\n"
            + (failure or
               f"  ours  = {got}\n  numpy = {reference(w, act)}"))


# ---------------------------------------------------------------------------
# The harness itself
# ---------------------------------------------------------------------------

def test_a_wrong_answer_would_actually_fail(cim_opt, cim_run):
    """Guards against the whole file passing because nothing is compared.

    Every test above compares the pipeline to `reference`. If `reference`
    silently returned the pipeline's own output -- or if parse_output
    returned an empty array that compares equal to nothing -- they would all
    pass while testing nothing. Perturbing one weight must change the
    answer.
    """
    w = np.eye(4, dtype=np.int8)
    act = np.array([1, 2, 3, 4], np.int8)
    got, _ = compile_and_run(cim_opt, cim_run, w, act)

    assert got.size == 4, "the output was not read back at all"
    assert np.array_equal(got, act.astype(np.int64))
    assert not np.array_equal(got, reference(w + 1, act)), (
        "changing the weights did not change the reference answer")


def test_the_pipeline_reports_a_bad_target_instead_of_guessing(cim_opt):
    """A missing target file must fail loudly, not fall back to a default."""
    source = build_module(np.eye(4, dtype=np.int8),
                          np.array([1, 2, 3, 4], np.int8))
    result = run([cim_opt, "-", "--cim-detect",
                  "--cim-partition=target-yaml=targets/not-a-real-target.yaml"],
                 stdin=source)
    assert result.returncode != 0, (
        "cim-opt accepted a nonexistent target file; every cost number it "
        "then produced would be against an imaginary device")
