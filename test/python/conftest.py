"""Shared fixtures for the differential tests.

These tests compare our C++ implementations against independent Python
implementations of the same contract:

  * test_yaml_differential.py   -- our target reader vs PyYAML
  * test_numerical_differential.py -- the compiled pipeline vs numpy

Both drive real binaries as subprocesses. That is the point: an in-process
C++ test can only compare our code against our own understanding of it,
while a subprocess test compares the shipped artifact against a tool that
was written by other people for other reasons. The two find different bugs.

Binaries are located by searching the usual build directories rather than
being configured in, so `pytest test/python` works straight from a checkout
after any of the documented build recipes. A missing binary skips the
tests that need it instead of failing -- the core-only build legitimately
has no cim-opt.
"""

import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# Searched in order. CIM_BUILD_DIR wins so CI can point at a build tree with
# a name we do not predict here.
_BUILD_DIRS = [
    os.environ.get("CIM_BUILD_DIR", ""),
    os.path.join(REPO_ROOT, "build"),
    os.path.join(REPO_ROOT, "build-cov"),
    os.path.join(REPO_ROOT, "build-asan"),
]


def find_tool(name):
    """Absolute path to a built tool, or None if it was not built."""
    for build in _BUILD_DIRS:
        if not build:
            continue
        candidate = os.path.join(build, "bin", name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which(name)


def require_tool(name):
    path = find_tool(name)
    if path is None:
        pytest.skip(f"{name} was not built; searched {_BUILD_DIRS}")
    return path


def run(argv, stdin=None, timeout=120):
    """Run a subprocess, returning CompletedProcess with text streams."""
    return subprocess.run(
        argv,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def cim_bench():
    return require_tool("cim-bench")


@pytest.fixture(scope="session")
def cim_opt():
    return require_tool("cim-opt")


@pytest.fixture(scope="session")
def cim_run():
    return require_tool("cim-run")
