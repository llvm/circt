#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pytest harness for the SerializationProbes integration test.

This builds the C++ driver under ``sw/serialization_probes.cpp`` against
generated ESI facade headers and runs it against a cosim-driven instance of
``hw/serialization_probes.py``. Each probe asserts an exact, position-revealing
result so any drift in the host serializer or deserializer (vs hardware)
fails loudly.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

from esiaccel.cosim.pytest import cosim_test

from .conftest import (HW_DIR, SW_DIR, check_lines, get_runtime_root,
                       require_tool)

PROBES_EXPECTED = [
    "byte_rotate1 ok",
    "byte_pattern_const ok",
    "byte_pattern_echo_eq ok",
    "sign_probe ok",
    "sign_probe13 ok",
    "pack_probe ok",
    "bit_pack_probe ok",
    "array_probe ok",
]


@cosim_test(HW_DIR / "serialization_probes.py")
class TestSerializationProbes:
  """End-to-end serialization-correctness probes."""

  def test_serialization_probes_cpp(self, tmp_path: Path, host: str, port: int,
                                    sources_dir: Path) -> None:
    require_tool("cmake")

    runtime_root = get_runtime_root()

    include_dir = tmp_path / "include"
    generated_dir = include_dir / "serialization_probes"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Codegen was already run automatically by cosim_test; copy the generated
    # headers into the per-test include tree.
    codegen_src = sources_dir / "generated"
    if codegen_src.exists():
      for item in codegen_src.iterdir():
        if item.is_file():
          shutil.copy(item, generated_dir)

    build_dir = tmp_path / "serialization_probes-build"
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
        [
            "cmake", "--build",
            str(build_dir), "--target", "serialization_probes_test"
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cmake build failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")

    result = subprocess.run(
        [
            str(build_dir / "serialization_probes_test"), "cosim",
            f"{host}:{port}"
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"serialization_probes_test failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")
    check_lines(result.stdout, PROBES_EXPECTED)
