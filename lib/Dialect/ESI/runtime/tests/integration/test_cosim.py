#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Optional, Sequence

import pytest

from esiaccel.cosim.pytest import cosim_test

ROOT_DIR = Path(__file__).resolve().parent
HW_DIR = ROOT_DIR / "hw"
SW_DIR = ROOT_DIR / "sw"


@contextlib.contextmanager
def temp_env(overrides: Dict[str, Optional[str]]):
  original: Dict[str, Optional[str]] = {}
  for key, value in overrides.items():
    original[key] = os.environ.get(key)
    if value is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = value
  try:
    yield
  finally:
    for key, value in original.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value


def require_tool(tool: str) -> None:
  if shutil.which(tool) is None:
    pytest.skip(f"Required tool not found in PATH: {tool}")


def require_runtime_root() -> Path:
  from esiaccel.utils import get_runtime_root
  return get_runtime_root()


def run_hw_script(script: Path, output_dir: Path, args: Sequence[str]) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  subprocess.run(
      [sys.executable, str(script),
       str(output_dir), *args],
      check=True,
      cwd=output_dir,
  )


def run_sw_script(script: Path,
                  host: str,
                  port: int,
                  env: Optional[Dict[str, Optional[str]]] = None) -> None:
  cmd = [sys.executable, str(script), "cosim", f"{host}:{port}"]
  if env:
    with temp_env(env):
      subprocess.run(cmd, check=True)
  else:
    subprocess.run(cmd, check=True)


@cosim_test(HW_DIR / "esi_test.py")
def test_cosim_esi_test(host: str, port: int) -> None:
  run_sw_script(SW_DIR / "esi_test.py", host, port)


@cosim_test(HW_DIR / "esi_test.py")
def test_cosim_esi_test_manifest_mmio(host: str, port: int) -> None:
  run_sw_script(SW_DIR / "esi_test.py", host, port,
                {"ESI_COSIM_MANIFEST_MMIO": "1"})


@cosim_test(HW_DIR / "esi_advanced.py")
def test_cosim_esi_advanced(host: str, port: int) -> None:
  run_sw_script(SW_DIR / "esi_advanced.py", host, port)


@cosim_test(HW_DIR / "esi_ram.py")
def test_cosim_esi_ram(host: str, port: int) -> None:
  run_sw_script(SW_DIR / "esi_ram.py", host, port)


@cosim_test(HW_DIR / "loopback.py")
def test_cosim_loopback(host: str, port: int) -> None:
  run_sw_script(SW_DIR / "loopback.py", host, port)


@cosim_test(HW_DIR / "esitester.py", args=("{tmp_dir}", "cosim"))
def test_cosim_esitester(host: str, port: int) -> None:
  require_tool("esitester")
  require_tool("esiquery")
  conn = f"{host}:{port}"
  commands = [
    ["esitester", "-v", "cosim", conn, "callback", "-i", "5"],
    ["esitester", "cosim", conn, "streaming_add"],
    ["esitester", "cosim", conn, "streaming_add", "-t"],
    ["esitester", "cosim", conn, "translate_coords"],
    ["esitester", "cosim", conn, "serial_coords", "-n", "40", "-b", "33"],
    ["esiquery", "cosim", conn, "telemetry"],
  ]
  for cmd in commands:
    subprocess.run(cmd, check=True)


@cosim_test(HW_DIR / "esitester.py", args=("{tmp_dir}", "cosim_dma"))
def test_cosim_esitester_dma(host: str, port: int) -> None:
  require_tool("esitester")
  require_tool("esiquery")
  conn = f"{host}:{port}"
  commands = [
    ["esitester", "cosim", conn, "hostmem"],
    ["esitester", "cosim", conn, "dma", "-w", "-r"],
    ["esiquery", "cosim", conn, "telemetry"],
  ]
  for cmd in commands:
    subprocess.run(cmd, check=True)


@cosim_test(HW_DIR / "loopback.py")
@pytest.mark.parametrize("mode", ["from_manifest", "from_accel"])
def test_loopback_cpp_codegen(mode: str, tmp_path: Path, host: str,
                port: int, sources_dir: Path) -> None:
  require_tool("cmake")

  runtime_root = require_runtime_root()

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

  build_dir = tmp_path / f"loopback-build-{mode}"
  subprocess.run(
      [
          "cmake",
          "-S",
          str(SW_DIR),
          "-B",
          str(build_dir),
          f"-DLOOPBACK_GENERATED_DIR={include_dir}",
          f"-DESI_RUNTIME_ROOT={runtime_root}",
      ],
      check=True,
  )
  subprocess.run(
      ["cmake", "--build",
       str(build_dir), "--target", "loopback_test"],
      check=True,
  )

  subprocess.run([str(build_dir / "loopback_test"), "cosim", f"{host}:{port}"],
                 check=True)
