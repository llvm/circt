#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Pytest wrapper that exposes each GoogleTest case inside ESIRuntimeCppTests as
an individual pytest item.

The binary is located by (in order):
  1. The ``ESI_RUNTIME_TESTS_BIN`` environment variable (explicit override).
  2. ``tests/cpp/ESIRuntimeCppTests`` relative to the ESI runtime root, which
     is derived from the ``esiaccel`` package location — the same convention
     used by the integration tests (three ``parent`` levels up from
     ``esiaccel.__file__``).

The entire module is skipped when the binary cannot be found.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import get_runtime_root

# ---------------------------------------------------------------------------
# Locate the test binary
# ---------------------------------------------------------------------------

_BIN_NAME = "ESIRuntimeCppTests" + (".exe" if sys.platform == "win32" else "")


def _find_binary() -> Path | None:
  # 1. Explicit override.
  env = os.environ.get("ESI_RUNTIME_TESTS_BIN")
  if env:
    p = Path(env)
    if p.is_file():
      return p

  # 2. Relative to the ESI runtime root derived from the esiaccel package —
  #    the same convention used by the integration tests.  Works when the
  #    standalone runtime build's esiaccel is the active one on PYTHONPATH.
  try:
    candidate = get_runtime_root() / "tests" / "cpp" / _BIN_NAME
    if candidate.is_file():
      return candidate
  except ImportError:
    pass

  return None


_BINARY = _find_binary()

if _BINARY is None:
  pytest.skip(
      "ESIRuntimeCppTests binary not found – make sure you've built that "
      "target and/or set ESI_RUNTIME_TESTS_BIN",
      allow_module_level=True,
  )

# ---------------------------------------------------------------------------
# Enumerate gtest cases
# ---------------------------------------------------------------------------


def _list_tests() -> list[str]:
  """Return a list of 'Suite.TestName' strings from --gtest_list_tests."""
  result = subprocess.run(
      [str(_BINARY), "--gtest_list_tests"],
      capture_output=True,
      text=True,
  )
  if result.returncode != 0:
    pytest.fail(
        f"Failed to list gtest cases (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}",
        pytrace=False,
    )
  tests: list[str] = []
  suite = ""
  for line in result.stdout.splitlines():
    # Suite header line ends with '.', e.g. "ESITypesTest."
    # Use a non-indent check so parameterized suites like "Suite/0." also match.
    if not line.startswith(" ") and line.rstrip().endswith("."):
      suite = line.strip()
    # Individual test line is indented, e.g. "  VoidTypeSerialization"
    elif line.startswith("  "):
      test_name = line.strip().split()[0]  # strip trailing comments
      if suite and test_name:
        tests.append(f"{suite}{test_name}")
  return tests


_ALL_TESTS = _list_tests()

# ---------------------------------------------------------------------------
# Parametrize
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
  if "gtest_case" in metafunc.fixturenames:
    metafunc.parametrize("gtest_case", _ALL_TESTS, ids=_ALL_TESTS)


def test_cpp_runtime(gtest_case: str) -> None:
  result = subprocess.run(
      [str(_BINARY), f"--gtest_filter={gtest_case}"],
      capture_output=True,
      text=True,
  )
  if result.returncode != 0:
    pytest.fail(
        f"gtest case '{gtest_case}' failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}",
        pytrace=False,
    )
