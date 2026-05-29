#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

# The Trace backend uses ';' as the connection-string separator on Windows.
_TRACE_SEP = ";" if os.name == "nt" else ":"

import pytest

from esiaccel.cosim.pytest import cosim_test

from .conftest import (HW_DIR, SW_DIR, build_cpp_test, check_lines,
                       get_runtime_root, require_tool, run_probe)

LOOPBACK_PROBES = [
    ("depth_constant", ["depth: 0x5", "depth_constant ok"]),
    ("loopback_i8", ["loopback_i8 ok: 0x5a"]),
    ("struct_func", ["struct_func ok: b=-7 x=-6 y=-7"]),
    ("odd_struct_func",
     ["odd_struct_func ok: a=2749 b=-20 p=10 q=-5 r0=4 r1=6"]),
    ("array_func", ["array_func ok: -3 -2"]),
    ("serial_coord_translate", ["serial_coord_translate ok"]),
]

LOOPBACK_TYPED_PROBES = [
    ("depth_constant", ["depth: 0x5", "depth_constant ok"]),
    ("loopback_i8", ["loopback_i8 ok: 0x5a"]),
    ("sint4_loopback", ["sint4_loopback ok: pos=5 neg=-3"]),
    ("struct_func", ["struct_func ok: b=-7 x=-6 y=-7"]),
    ("odd_struct_func",
     ["odd_struct_func ok: a=2749 b=-20 p=10 q=-5 r0=4 r1=6"]),
    ("array_func", ["array_func ok: -3 -2"]),
    ("serial_coord_translate", ["serial_coord_translate ok"]),
]


def _build_loopback_codegen(tmp_path: Path, host: str, port: int) -> Path:
  """Run live-connection codegen and build the loopback_test binary.

  This test deliberately uses ``esiaccel.codegen --platform cosim`` (live
  connection) rather than ``--file`` (offline manifest) to exercise that
  codegen path.
  """
  require_tool("cmake")
  require_tool("ninja")

  runtime_root = get_runtime_root()

  include_dir = tmp_path / "include"
  generated_dir = include_dir / "loopback"
  generated_dir.mkdir(parents=True, exist_ok=True)

  subprocess.run(
      [
          sys.executable,
          "-m",
          "esiaccel.codegen",
          "--platform",
          "cosim",
          "--connection",
          f"{host}:{port}",
          "--output-dir",
          str(generated_dir),
      ],
      check=True,
  )

  # Verify generated header content (LOOPBACK-H checks).
  header_path = generated_dir / "LoopbackIP.h"
  assert header_path.exists(), "Generated header LoopbackIP.h not found"
  header_content = header_path.read_text()
  check_lines(header_content, [
      "/// Generated header for esi_system module LoopbackIP.",
      "#pragma once",
      '#include "types.h"',
      "namespace esi_system {",
      "class LoopbackIP {",
      "static constexpr uint32_t depth = 0x5;",
      "} // namespace esi_system",
  ])

  build_dir = tmp_path / "loopback-build"
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
       str(build_dir), "--target", "loopback_test"],
      capture_output=True,
      text=True,
  )
  assert result.returncode == 0, (
      f"cmake build failed (rc={result.returncode}):\n"
      f"--- stdout ---\n{result.stdout}\n"
      f"--- stderr ---\n{result.stderr}")

  return build_dir / "loopback_test"


@cosim_test(HW_DIR / "loopback.py")
class TestLoopback:
  """Tests for esiquery against the loopback design."""

  def test_loopback_cpp_codegen(self, tmp_path: Path, host: str,
                                port: int) -> None:
    """Build against live-connection codegen and run all probes."""
    binary = _build_loopback_codegen(tmp_path, host, port)
    for probe, expected in LOOPBACK_PROBES:
      run_probe(binary, host, port, probe, expected)

  def test_loopback_typed_cpp_codegen(self, host: str, port: int,
                                      sources_dir: Path) -> None:
    binary = build_cpp_test(sources_dir, "loopback_typed_test", "loopback")
    for probe, expected in LOOPBACK_TYPED_PROBES:
      run_probe(binary, host, port, probe, expected)

  def test_loopback_query_info(self, sources_dir: Path) -> None:
    """Verify esiquery info output against the generated manifest
    (QUERY-INFO checks)."""
    require_tool("esiquery")
    manifest = sources_dir / "esi_system_manifest.json"
    assert manifest.exists(), "Manifest not found"
    result = subprocess.run(
        ["esiquery", "trace", f"w{_TRACE_SEP}{manifest}", "info"],
        check=True,
        capture_output=True,
        text=True,
    )
    check_lines(result.stdout, [
        "API version: 0",
        "* Module information",
        "- LoopbackIP v0.0",
        "IP which simply echos bytes",
        "Constants:",
        "depth: 5",
        "Extra metadata:",
        "foo: 1",
    ])

  def test_loopback_query_hier(self, sources_dir: Path) -> None:
    """Verify esiquery hier output against the generated manifest
    (QUERY-HIER checks)."""
    require_tool("esiquery")
    manifest = sources_dir / "esi_system_manifest.json"
    assert manifest.exists(), "Manifest not found"
    result = subprocess.run(
        ["esiquery", "trace", f"w{_TRACE_SEP}{manifest}", "hier"],
        check=True,
        capture_output=True,
        text=True,
    )
    check_lines(result.stdout, [
        "* Design hierarchy",
        "func1: function uint16(uint16)",
        "structFunc: function ResultStruct(ArgStruct)",
        "arrayFunc: function ResultArray(sint8[1])",
        "* Instance: loopback_inst[0]",
        "loopback_tohw:",
        "recv: bits8",
        "loopback_fromhw:",
        "send: bits8",
        "mysvc_recv:",
        "recv: void",
        "mysvc_send:",
        "send: void",
        "* Instance: loopback_inst[1]",
    ])
