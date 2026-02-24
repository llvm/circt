#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from esiaccel.cosim.pytest import cosim_test

from .conftest import HW_DIR, SW_DIR, check_lines, require_tool


@cosim_test(HW_DIR / "loopback.py")
@pytest.mark.parametrize("mode", ["from_manifest", "from_accel"])
def test_loopback_cpp_codegen(mode: str, tmp_path: Path, host: str, port: int,
                              sources_dir: Path) -> None:
  require_tool("cmake")

  from esiaccel.utils import get_runtime_root
  runtime_root = get_runtime_root()

  include_dir = tmp_path / "include"
  generated_dir = include_dir / "loopback"
  generated_dir.mkdir(parents=True, exist_ok=True)
  if mode == "from_manifest":
    # Codegen was already run automatically; copy the generated code.
    codegen_src = sources_dir / "generated"
    if codegen_src.exists():
      for item in codegen_src.iterdir():
        if item.is_file():
          shutil.copy(item, generated_dir)
  else:
    # Generate from live cosim connection instead.
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

  build_dir = tmp_path / f"loopback-build-{mode}"
  result = subprocess.run(
      [
          "cmake",
          "-S",
          str(SW_DIR),
          "-B",
          str(build_dir),
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

  # Run the C++ test binary and verify output (CPP-TEST checks).
  result = subprocess.run(
      [str(build_dir / "loopback_test"), "cosim", f"{host}:{port}"],
      check=True,
      capture_output=True,
      text=True,
  )
  check_lines(result.stdout, [
      "depth: 0x5",
      "loopback i8 ok: 0x5a",
      "struct func ok: b=-7 x=-6 y=-7",
      "odd struct func ok: a=2749 b=-20 p=10 q=-5 r0=4 r1=6",
      "array func ok: -3 -2",
  ])


@cosim_test(HW_DIR / "loopback.py")
class TestLoopbackQuery:
  """Tests for esiquery against the loopback design."""

  def test_loopback_query_info(self, sources_dir: Path) -> None:
    """Verify esiquery info output against the generated manifest
    (QUERY-INFO checks)."""
    require_tool("esiquery")
    manifest = sources_dir / "esi_system_manifest.json"
    assert manifest.exists(), "Manifest not found"
    result = subprocess.run(
        ["esiquery", "trace", f"w:{manifest}", "info"],
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
        ["esiquery", "trace", f"w:{manifest}", "hier"],
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
