#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
from pathlib import Path
import logging
import shutil
import subprocess
from typing import Optional

import pytest

_logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
HW_DIR = ROOT_DIR / "hw"
SW_DIR = ROOT_DIR / "sw"

from tests.conftest import get_runtime_root  # noqa: F401 – re-exported


def require_tool(tool: str) -> None:
  if shutil.which(tool) is None:
    pytest.skip(f"Required tool not found in PATH: {tool}")


def require_env(var_name: str) -> str:
  value = os.environ.get(var_name)
  if not value:
    pytest.skip(f"Required environment variable not set: {var_name}")
  return value


def run_cmd(cmd, **kwargs) -> str:
  """Run a command, capture stdout, and return it. Raises on failure."""
  _logger.info("run_cmd: %s", cmd)
  result = subprocess.run(cmd,
                          check=True,
                          capture_output=True,
                          text=True,
                          **kwargs)
  _logger.debug("stdout: %s", result.stdout)
  _logger.debug("stderr: %s", result.stderr)
  return result.stdout


def check_lines(stdout: str, expected: list[str]) -> None:
  """Assert that every expected substring appears in stdout in order."""
  remaining = stdout
  for line in expected:
    idx = remaining.find(line)
    assert idx >= 0, \
        f"Expected output not found: {line!r}"
    remaining = remaining[idx + len(line):]


def _runtime_env() -> dict[str, str]:
  """Return an environment which can load ESI runtime shared libraries."""
  env = os.environ.copy()
  if os.name != "nt":
    return env

  runtime_root = get_runtime_root()
  dll_dirs = [
      path for path in (runtime_root, runtime_root / "bin",
                        runtime_root / "lib") if path.exists()
  ]
  existing_path = env.get("PATH", "")
  env["PATH"] = os.pathsep.join(str(path) for path in dll_dirs) + \
      os.pathsep + existing_path
  return env


# ---------------------------------------------------------------------------
# Shared C++ build helper
# ---------------------------------------------------------------------------


def build_cpp_test(sources_dir: Path,
                   target: str,
                   header_subdir: str,
                   build_subdir: Optional[str] = None) -> Path:
  """Configure + build a C++ integration test target, returning the binary.

  * ``sources_dir``: root provided by ``cosim_test`` (contains ``generated/``).
  * ``target``: CMake target name (e.g. ``loopback_test``).
  * ``header_subdir``: subdirectory under the include root where the generated
    headers are copied (e.g. ``"loopback"`` or ``"test_codegen"``).
  * ``build_subdir``: name for the build directory under ``sources_dir``.
    Defaults to ``target``.

  The configure step is skipped when the build directory already exists;
  ``cmake --build`` always runs so that CMake's own dependency tracking
  picks up any source or generated-header changes.
  """
  require_tool("cmake")
  require_tool("ninja")

  if build_subdir is None:
    build_subdir = target
  build_dir = sources_dir / build_subdir
  exe_suffix = ".exe" if os.name == "nt" else ""
  binary = build_dir / (target + exe_suffix)

  runtime_root = get_runtime_root()
  include_dir = sources_dir / "cpp_include"
  generated_dir = include_dir / header_subdir

  if not build_dir.exists():
    generated_dir.mkdir(parents=True, exist_ok=True)

    codegen_src = sources_dir / "generated"
    if codegen_src.exists():
      for item in codegen_src.iterdir():
        if item.is_file():
          shutil.copy(item, generated_dir)

    result = subprocess.run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(SW_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DLOOPBACK_GENERATED_DIR={include_dir}",
            f"-DESI_RUNTIME_ROOT={runtime_root}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cmake configure failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")

  result = subprocess.run(
      ["cmake", "--build",
       str(build_dir), "--target", target],
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, (
      f"cmake build failed (rc={result.returncode}):\n"
      f"--- stdout ---\n{result.stdout}\n"
      f"--- stderr ---\n{result.stderr}")

  return binary


def run_probe(binary: Path,
              host: str,
              port: int,
              probe: str,
              expected: Optional[list[str]] = None) -> str:
  """Invoke ``binary --probe <probe>`` and assert it prints ``<probe> ok``.

  If *expected* is given, those substrings are checked (in order) against
  stdout instead of the default ``["<probe> ok"]``.

  Returns the captured stdout for further assertions if needed.
  """
  result = subprocess.run(
      [str(binary), "--probe", probe, "cosim", f"{host}:{port}"],
      capture_output=True,
      env=_runtime_env(),
      text=True,
      timeout=60,
  )
  assert result.returncode == 0, (
      f"{binary.name} --probe {probe} failed (rc={result.returncode}):\n"
      f"--- stdout ---\n{result.stdout}\n"
      f"--- stderr ---\n{result.stderr}")
  check_lines(result.stdout, expected or [f"{probe} ok"])
  return result.stdout
